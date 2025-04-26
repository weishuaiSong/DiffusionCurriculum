from dataclasses import asdict, dataclass, field
from collections import defaultdict
import contextlib
import logging
import os
import datetime
from concurrent import futures
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.loaders.utils import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.schedulers import UNet2DConditionModel
import numpy as np
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from train.trainer.common.pipeline_with_logprob import pipeline_with_logprob
from train.trainer.common.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
import torch.distributions.kl as kl

t = partial(tqdm.tqdm, dynamic_ncols=True)

logger = logging.getLogger(__name__)


@dataclass
class Config:
    resume_from: str | None = field(default=None)
    num_epochs: int = field(default=100)
    sample_batch_size: int = field(default=16)
    train_batch_size: int = field(default=16)
    gradient_accumulation_steps: int = field(default=1)
    num_inner_epochs: int = field(default=1)
    sample_num_step: int = field(default=50)
    train_num_step: int = field(default=1000)
    timestep_fraction: float = field(default=0.8)
    log_dir: str = field(default="logs")
    num_checkpoint_limit: int = field(default=10)
    seed: int | None = field(default=None)
    use_lora: bool = field(default=False)
    sd_model: str = field(default="CompVis/stable-diffusion-v1-4")
    sd_revision: str | None = field(default=None)
    learning_rate: float = field(default=1e-4)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_weight_decay: float = field(default=0.01)
    adam_epsilon: float = field(default=1e-8)
    sample_num_batches_per_epoch: int = field(default=10)


class Trainer:
    def __init__(self, config: Config) -> None:
        self.config = config
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        self.run_name = f"dpok_{unique_id}"
        self.accelerator: Accelerator | None = None
        if self.config.resume_from:
            self.config.resume_from = self._norm_path(self.config.resume_from)

        self._init()

    def step(self): ...

    def _fix_seed(self):
        assert self.accelerator, "should call after init accelerator"
        # set seed (device_specific is very important to get different prompts on different devices)
        np.random.seed(self.config.seed or 114514)
        available_devices = self.accelerator.num_processes
        random_seeds = np.random.randint(0, 100000, size=available_devices)
        device_seed = random_seeds[self.accelerator.process_index]  # type: ignore
        set_seed(device_seed, device_specific=True)

    def _norm_path(self, path: str) -> str:
        res = os.path.normpath(os.path.expanduser(path))
        if "checkpoint_" not in os.path.basename(path):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(self.config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {path}")
            res = os.path.join(
                path,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        return res

    def _init(self):
        # number of timesteps within each trajectory to train on
        num_train_timesteps = int(self.config.sample_num_step * self.config.timestep_fraction)

        accelerator_config = ProjectConfiguration(
            project_dir=os.path.join(self.config.log_dir, self.run_name),
            automatic_checkpoint_naming=True,
            total_limit=self.config.num_checkpoint_limit,
        )

        self.accelerator = Accelerator(
            log_with="wandb",
            project_config=accelerator_config,
            # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
            # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
            # the total number of optimizer steps to accumulate across.
            gradient_accumulation_steps=self.config.gradient_accumulation_steps * num_train_timesteps,
        )
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name="ddpo-pytorch",
                config=asdict(self.config),
                init_kwargs={"wandb": {"name": self.run_name}},
            )
        logger.info(f"\n{self.config}")

        # load scheduler, tokenizer and models.
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.sd_model, revision=self.config.sd_revision
        )
        # freeze parameters of models to save more memory
        self.sd_pipeline.vae.requires_grad_(False)
        self.sd_pipeline.text_encoder.requires_grad_(False)
        self.sd_pipeline.unet.requires_grad_(not self.config.use_lora)
        # disable safety checker
        self.sd_pipeline.safety_checker = None
        # make the progress bar nicer
        self.sd_pipeline.set_progress_bar_config(
            position=1,
            disable=not self.accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )
        # switch to DDIM scheduler
        self.sd_pipeline.scheduler = DDIMScheduler.from_config(self.sd_pipeline.scheduler.config)

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        inference_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to inference_dtype
        self.sd_pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.text_encoder.to(self.accelerator.device, dtype=inference_dtype)
        if self.config.use_lora:
            self.sd_pipeline.unet.to(self.accelerator.device, dtype=inference_dtype)

        if self.config.use_lora:
            # Set correct lora layers
            lora_attn_procs = {}
            for name in self.sd_pipeline.unet.attn_processors.keys():
                cross_attention_dim = (
                    None if name.endswith("attn1.processor") else self.sd_pipeline.unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = self.sd_pipeline.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.sd_pipeline.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.sd_pipeline.unet.config.block_out_channels[block_id]

                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim  # type: ignore
                )
            self.sd_pipeline.unet.set_attn_processor(lora_attn_procs)
            trainable_layers = AttnProcsLayers(self.sd_pipeline.unet.attn_processors)
        else:
            trainable_layers = self.sd_pipeline.unet

        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)

        optimizer = torch.optim.AdamW(
            trainable_layers.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )

        # prepare prompt and reward fn
        prompt_fn = getattr(d3po_pytorch.prompts, config.prompt_fn)
        reward_fn = getattr(d3po_pytorch.rewards, config.reward_fn)()

        # generate negative prompt embeddings
        neg_prompt_embed = self.sd_pipeline.text_encoder(
            self.sd_pipeline.tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
        )[0]
        sample_neg_prompt_embeds = neg_prompt_embed.repeat(self.config.sample_batch_size, 1, 1)
        train_neg_prompt_embeds = neg_prompt_embed.repeat(self.config.train_batch_size, 1, 1)

        # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
        # more memory
        autocast = contextlib.nullcontext if self.config.use_lora else self.accelerator.autocast

        # Prepare everything with our `accelerator`.
        trainable_layers, optimizer = self.accelerator.prepare(trainable_layers, optimizer)

        # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
        # remote server running llava inference.
        self.executor = futures.ThreadPoolExecutor(max_workers=2)

        self.samples_per_epoch = (
            self.config.sample_batch_size * self.accelerator.num_processes * self.config.sample_num_batches_per_epoch
        )
        self.total_train_batch_size = (
            self.config.train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps
        )

        assert self.config.sample_batch_size >= self.config.train_batch_size
        assert self.config.sample_batch_size % self.config.train_batch_size == 0
        assert self.samples_per_epoch % self.total_train_batch_size == 0

        if self.config.resume_from:
            logger.info(f"Resuming from {self.config.resume_from}")
            self.accelerator.load_state(self.config.resume_from)
            self.first_epoch = int(self.config.resume_from.split("_")[-1]) + 1
        else:
            self.first_epoch = 0

    def _save_model_hook(self, models, weights, output_dir):
        assert len(models) == 1
        if self.config.use_lora and isinstance(models[0], AttnProcsLayers):
            self.sd_pipeline.unet.save_attn_procs(output_dir)
        elif not self.config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def _load_model_hook(self, models, input_dir):
        assert len(models) == 1
        if self.config.use_lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                self.config.sd_model, revision=self.config.sd_revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)  # type: ignore
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())  # type: ignore
            del tmp_unet
        elif not self.config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)  # type: ignore
            models[0].load_state_dict(load_model.state_dict())  # type: ignore
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    def train(self):
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {self.config.num_epochs}")
        logger.info(f"  Sample batch size per device = {self.config.sample_batch_size}")
        logger.info(f"  Train batch size per device = {self.config.train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info("")
        logger.info(f"  Total number of samples per epoch = {self.samples_per_epoch}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_train_batch_size}"
        )
        logger.info(
            f"  Number of gradient updates per inner epoch = {self.samples_per_epoch // self.total_train_batch_size}"
        )
        logger.info(f"  Number of inner epochs = {self.config.num_inner_epochs}")

    def epoch_loop(self):
        global_step = 0
        for epoch in range(self.first_epoch, self.config.num_epochs):
            #################### SAMPLING ####################
            self.sd_pipeline.unet.eval()
            samples = []
            prompts = []
            for i in t(
                range(config.sample.num_batches_per_epoch),
                desc=f"Epoch {epoch}: sampling",
                disable=not accelerator.is_local_main_process,
                position=0,
            ):
                # generate prompts
                prompts, prompt_metadata = zip(
                    *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
                )

                # encode prompts
                prompt_ids = pipeline.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length,
                ).input_ids.to(accelerator.device)
                prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

                # sample
                with autocast():
                    images, _, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        eta=config.sample.eta,
                        output_type="pt",
                    )

                latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
                log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
                timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)

                # compute rewards asynchronously
                rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
                # yield to to make sure reward computation starts
                eval_rewards = None
                if epoch % config.sample.eval_epoch == 0:
                    eval_prompts, eval_prompt_metadata = zip(
                        *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.eval_batch_size)]
                    )

                    # encode prompts
                    eval_prompt_ids = pipeline.tokenizer(
                        eval_prompts,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=pipeline.tokenizer.model_max_length,
                    ).input_ids.to(accelerator.device)
                    eval_prompt_embeds = pipeline.text_encoder(eval_prompt_ids)[0]
                    eval_sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.eval_batch_size, 1, 1)
                    # sample
                    with autocast():
                        eval_images, _, _, _ = pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=eval_prompt_embeds,
                            negative_prompt_embeds=eval_sample_neg_prompt_embeds,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            eta=config.sample.eta,
                            output_type="pt",
                        )

                    # compute rewards asynchronously
                    eval_rewards = executor.submit(reward_fn, eval_images, eval_prompts, eval_prompt_metadata)

                samples.append(
                    {
                        "prompt_ids": prompt_ids,
                        "prompt_embeds": prompt_embeds,
                        "timesteps": timesteps,
                        "latents": latents[:, :-1],  # each entry is the latent before timestep t
                        "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                        "log_probs": log_probs,
                        "rewards": rewards,
                        "eval_rewards": eval_rewards,
                    }
                )

            # wait for all rewards to be computed
            for sample in t(
                samples,
                desc="Waiting for rewards",
                disable=not accelerator.is_local_main_process,
                position=0,
            ):
                rewards, reward_metadata = sample["rewards"].result()
                # accelerator.print(reward_metadata)
                sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)
                if sample["eval_rewards"] is not None:
                    eval_rewards, eval_reward_metadata = sample["eval_rewards"].result()
                    sample["eval_rewards"] = torch.as_tensor(rewards, device=accelerator.device)
                else:
                    del sample["eval_rewards"]

            # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
            samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

            # gather rewards across processes
            rewards = accelerator.gather(samples["rewards"]).cpu().numpy()

            # log rewards and images
            if samples.get("eval_rewards") is not None:
                eval_rewards = accelerator.gather(samples["eval_rewards"]).cpu().numpy()
                accelerator.log(
                    {
                        "reward": eval_rewards,
                        "num_samples": epoch * available_devices * config.sample.batch_size,
                        "reward_mean": eval_rewards.mean(),
                        "reward_std": eval_rewards.std(),
                    },
                    step=global_step,
                )
            else:
                accelerator.log(
                    {
                        "reward": rewards,
                        "num_samples": epoch * available_devices * config.sample.batch_size,
                        "reward_mean": rewards.mean(),
                        "reward_std": rewards.std(),
                    },
                    step=global_step,
                )
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                for i, image in enumerate(images):
                    pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((256, 256))
                    pil.save(os.path.join(tmpdir, f"{i}.jpg"))
                accelerator.log(
                    {
                        "images": [
                            wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt:.25} | {reward:.2f}")
                            for i, (prompt, reward) in enumerate(zip(prompts, rewards))
                        ],
                    },
                    step=global_step,
                )

            # per-prompt mean/std tracking
            if config.per_prompt_stat_tracking:
                # gather the prompts across processes
                prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
                prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                advantages = stat_tracker.update(prompts, rewards)
            else:
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # ungather advantages; we only need to keep the entries corresponding to the samples on this process
            samples["advantages"] = (
                torch.as_tensor(advantages)
                .reshape(accelerator.num_processes, -1)[accelerator.process_index]
                .to(accelerator.device)
            )

            del samples["rewards"]
            del samples["prompt_ids"]

            total_batch_size, num_timesteps = samples["timesteps"].shape
            assert total_batch_size == config.sample.batch_size * config.sample.num_batches_per_epoch
            assert num_timesteps == config.sample.num_steps

            #################### TRAINING ####################
            for inner_epoch in range(config.train.num_inner_epochs):
                # shuffle samples along batch dimension
                perm = torch.randperm(total_batch_size, device=accelerator.device)
                samples = {k: v[perm] for k, v in samples.items()}

                # shuffle along time dimension independently for each sample
                perms = torch.stack(
                    [torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)]
                )
                for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                    samples[key] = samples[key][
                        torch.arange(total_batch_size, device=accelerator.device)[:, None], perms
                    ]

                # rebatch for training
                samples_batched = {k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) for k, v in samples.items()}

                # dict of lists -> list of dicts for easier iteration
                samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

                # train
                pipeline.unet.train()
                info = defaultdict(list)
                for i, sample in t(
                    list(enumerate(samples_batched)),
                    desc=f"Epoch {epoch}.{inner_epoch}: training",
                    position=0,
                    disable=not accelerator.is_local_main_process,
                ):
                    if config.train.cfg:
                        # concat negative prompts to sample prompts to avoid two forward passes
                        embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
                    else:
                        embeds = sample["prompt_embeds"]

                    for j in t(
                        range(num_train_timesteps),
                        desc="Timestep",
                        position=1,
                        leave=False,
                        disable=not accelerator.is_local_main_process,
                    ):
                        with accelerator.accumulate(pipeline.unet):
                            with autocast():
                                if config.train.cfg:
                                    noise_pred = pipeline.unet(
                                        torch.cat([sample["latents"][:, j]] * 2),
                                        torch.cat([sample["timesteps"][:, j]] * 2),
                                        embeds,
                                    ).sample
                                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                    noise_pred = noise_pred_uncond + config.sample.guidance_scale * (
                                        noise_pred_text - noise_pred_uncond
                                    )
                                else:
                                    noise_pred = pipeline.unet(
                                        sample["latents"][:, j], sample["timesteps"][:, j], embeds
                                    ).sample
                                # compute the log prob of next_latents given latents under the current model
                                _, log_prob = ddim_step_with_logprob(
                                    pipeline.scheduler,
                                    noise_pred,
                                    sample["timesteps"][:, j],
                                    sample["latents"][:, j],
                                    eta=config.sample.eta,
                                    prev_sample=sample["next_latents"][:, j],
                                )

                            # ppo logic
                            advantages = torch.clamp(
                                sample["advantages"], -config.train.adv_clip_max, config.train.adv_clip_max
                            )
                            ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                            unclipped_loss = -advantages * ratio
                            clipped_loss = -advantages * torch.clamp(
                                ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range
                            )
                            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                            kl_divergence = kl.kl_divergence(
                                torch.distributions.Categorical(logits=log_prob),
                                torch.distributions.Categorical(logits=sample["log_probs"][:, j]),
                            )
                            loss += config.kl_ratio * kl_divergence.mean()
                            # debugging values
                            # John Schulman says that (ratio - 1) - log(ratio) is a better
                            # estimator, but most existing code uses this so...
                            # http://joschu.net/blog/kl-approx.html
                            info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                            info["clipfrac"].append(
                                torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float())
                            )
                            info["loss"].append(loss)

                            # backward pass
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(trainable_layers.parameters(), config.train.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()

                        # Checks if the accelerator has performed an optimization step behind the scenes
                        if accelerator.sync_gradients:
                            assert (j == num_train_timesteps - 1) and (
                                i + 1
                            ) % config.train.gradient_accumulation_steps == 0
                            # log training-related stuff
                            info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                            info = accelerator.reduce(info, reduction="mean")
                            info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                            accelerator.log(info, step=global_step)
                            global_step += 1
                            info = defaultdict(list)

                # make sure we did an optimization step at the end of the inner epoch
                assert accelerator.sync_gradients

            if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
                accelerator.save_state()
