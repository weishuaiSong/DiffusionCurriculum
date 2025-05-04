from collections import defaultdict
import os
import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

import numpy as np
from train.curriculum import Curriculum
from train.trainer.common.state_tracker import PerPromptStatTracker
from train.trainer.common.pipeline_with_logprob import pipeline_with_logprob
from train.trainer.common.ddim_with_logprob import ddim_step_with_logprob
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Pipeline
from transformers.pipelines import pipeline
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image

t = partial(tqdm.tqdm, dynamic_ncols=True)

logger = get_logger(__name__)


class DDPODataset(Dataset):
    def __init__(
        self,
        samples: Dict[str, torch.Tensor],
        train_batch_size: int,
    ):
        """
        DDPO数据集类

        Args:
            samples: 采样得到的数据字典
            train_batch_size: 训练批次大小
        """
        self.samples = samples
        self.train_batch_size = train_batch_size
        self.total_samples = next(iter(samples.values())).shape[0]

        # 沿批次维度打乱样本
        perm = torch.randperm(self.total_samples)
        self.samples = {k: v[perm] for k, v in samples.items()}

        # 对每个样本独立地沿时间维度进行打乱
        total_batch_size, num_timesteps = self.samples["timesteps"].shape
        self.perms = torch.stack([torch.randperm(num_timesteps) for _ in range(total_batch_size)])
        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            self.samples[key] = self.samples[key][torch.arange(total_batch_size)[:, None], self.perms]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.samples.items()}


@dataclass
class Config:
    # 模型配置
    pretrained_model: str = field(default="runwayml/stable-diffusion-v1-5")
    pretrained_revision: str = field(default="main")

    # 随机种子
    seed: int = field(default=42)

    # 日志和检查点配置
    logdir: str = field(default="logs")
    run_name: str = field(default="")
    num_checkpoint_limit: int = field(default=10)
    save_freq: int = field(default=10)

    # 训练配置
    num_epochs: int = field(default=100)
    mixed_precision: str = field(default="bf16")
    allow_tf32: bool = field(default=True)
    resume_from: str = field(default="")

    # 采样配置
    sample_num_steps: int = field(default=50)
    sample_eta: float = field(default=0.0)
    sample_guidance_scale: float = field(default=5.0)
    sample_batch_size: int = field(default=4)
    sample_num_batches_per_epoch: int = field(default=4)
    sample_eval_batch_size: int = field(default=4)
    sample_eval_epoch: int = field(default=5)

    # 训练配置
    train_batch_size: int = field(default=1)
    train_learning_rate: float = field(default=1e-5)
    train_num_inner_epochs: int = field(default=1)
    train_gradient_accumulation_steps: int = field(default=1)
    train_max_grad_norm: float = field(default=1.0)
    train_cfg: bool = field(default=True)
    train_adv_clip_max: float = field(default=5.0)
    train_timestep_fraction: float = field(default=1.0)
    train_clip_range: float = field(default=1e-4)

    # Adam优化器配置
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_weight_decay: float = field(default=1e-4)
    adam_epsilon: float = field(default=1e-8)

    # 状态跟踪配置
    per_prompt_stat_tracking: bool = field(default=True)
    per_prompt_stat_tracking_buffer_size: int = field(default=16)
    per_prompt_stat_tracking_min_count: int = field(default=16)

    # 提示和奖励函数
    prompt_fn: str = field(default="simple_animals")
    prompt_fn_kwargs: dict = field(default_factory=dict)
    reward_fn: str = field(default="jpeg_compressibility")

    # 数据加载配置
    num_workers: int = field(default=4)
    dataloader_pin_memory: bool = field(default=True)


class Trainer:
    def __init__(
        self,
        curriculum: Curriculum,
        update_target_difficulty: Callable[[int], None],
        config: Config,
        reward_function: Callable[[Pipeline, torch.Tensor, tuple[str], tuple[Any]], torch.Tensor],
        prompt_function: Callable[[], tuple[str, Any]],
        vqa_model_name: str,
    ) -> None:
        self.curriculum = curriculum
        self.update_target_difficulty = update_target_difficulty
        self.config = config

        # 设置运行名称
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        if not self.config.run_name:
            self.config.run_name = unique_id
        else:
            self.config.run_name += "_" + unique_id

        if self.config.resume_from:
            self.config.resume_from = os.path.normpath(os.path.expanduser(self.config.resume_from))
            if "checkpoint_" not in os.path.basename(self.config.resume_from):
                # 获取此目录中最新的检查点
                checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(self.config.resume_from)))
                if len(checkpoints) == 0:
                    raise ValueError(f"No checkpoints found in {self.config.resume_from}")
                self.config.resume_from = os.path.join(
                    self.config.resume_from,
                    sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
                )

        # 每个轨迹中用于训练的时间步数
        self.num_train_timesteps = int(self.config.sample_num_steps * self.config.train_timestep_fraction)

        accelerator_config = ProjectConfiguration(
            project_dir=os.path.join(self.config.logdir, self.config.run_name),
            automatic_checkpoint_naming=True,
            total_limit=self.config.num_checkpoint_limit,
        )

        self.accelerator = Accelerator(
            log_with="wandb",
            mixed_precision=self.config.mixed_precision,
            project_config=accelerator_config,
            # 我们总是在时间步之间累积梯度；我们希望config.train.gradient_accumulation_steps是
            # 我们累积的*样本*数量，所以我们需要乘以训练时间步的数量来得到
            # 要跨累积的优化器步骤的总数。
            gradient_accumulation_steps=self.config.train_gradient_accumulation_steps * self.num_train_timesteps,
        )
        self.available_devices = self.accelerator.num_processes
        self._fix_seed()

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name="ddpo-pytorch",
                config=asdict(self.config),
                init_kwargs={"wandb": {"name": self.config.run_name}},
            )
        logger.info(f"\n{self.config}")

        # 加载调度器、分词器和模型
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.pretrained_model, revision=self.config.pretrained_revision, device_map="balanced", use_fast=True
        )
        # 冻结模型参数以节省更多内存
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.unet.requires_grad_(True)
        # 禁用安全检查器
        self.pipeline.safety_checker = None
        # 美化进度条
        self.pipeline.set_progress_bar_config(
            position=1,
            disable=not self.accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )
        # 切换到DDIM调度器
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)

        # 对于混合精度训练，我们将所有非训练权重（vae、非lora text_encoder和非lora unet）转换为半精度
        # 因为这些权重仅用于推理，因此无需保持全精度权重。
        inference_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16

        # 将unet、vae和text_encoder移至设备并转换为inference_dtype
        # self.pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        # self.pipeline.text_encoder.to(self.accelerator.device, dtype=inference_dtype)
        self.pipeline = self.accelerator.prepare(self.pipeline)
        self.vqa_pipeline = pipeline(
            "image-text-to-text",
            model=vqa_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            batch_size=config.train_batch_size,
        )

        self.vqa_pipeline.model.eval()
        for p in self.vqa_pipeline.model.parameters():
            p.requires_grad = False

        self.vqa_pipeline.model = self.accelerator.prepare(self.vqa_pipeline.model)

        self.trainable_layers = self.pipeline.unet

        # 设置使用Accelerate的diffusers友好的检查点保存
        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)

        # 为Ampere GPU启用TF32以加快训练速度，
        # 参见 https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # 初始化优化器
        optimizer_cls = torch.optim.AdamW

        self.optimizer = optimizer_cls(
            self.trainable_layers.parameters(),
            lr=self.config.train_learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )

        # 准备提示和奖励函数
        self.prompt_fn = prompt_function
        self.reward_fn = reward_function

        # 生成负面提示嵌入
        neg_prompt_embed = self.pipeline.text_encoder(
            self.pipeline.tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.pipeline.tokenizer.model_max_length,
            ).input_ids
        )[0]
        self.sample_neg_prompt_embeds = neg_prompt_embed.repeat(self.config.sample_batch_size, 1, 1)
        self.train_neg_prompt_embeds = neg_prompt_embed.repeat(self.config.train_batch_size, 1, 1)

        # 初始化统计跟踪器
        self.stat_tracker = None
        if self.config.per_prompt_stat_tracking:
            self.stat_tracker = PerPromptStatTracker(
                self.config.per_prompt_stat_tracking_buffer_size,
                self.config.per_prompt_stat_tracking_min_count,
            )

        # 出于某种原因，对于非lora训练autocast是必要的，但对于lora训练它不是必要的，而且会使用更多内存
        self.autocast = self.accelerator.autocast

        # 使用`accelerator`准备所有内容
        self.trainable_layers, self.optimizer = self.accelerator.prepare(self.trainable_layers, self.optimizer)

        # 计算每个epoch的样本数和批次大小
        self.samples_per_epoch = (
            self.config.sample_batch_size * self.accelerator.num_processes * self.config.sample_num_batches_per_epoch
        )
        self.total_train_batch_size = (
            self.config.train_batch_size
            * self.accelerator.num_processes
            * self.config.train_gradient_accumulation_steps
        )

        # 检查配置的一致性
        assert self.config.sample_batch_size >= self.config.train_batch_size
        assert self.config.sample_batch_size % self.config.train_batch_size == 0
        assert self.samples_per_epoch % self.total_train_batch_size == 0

        # 如果从检查点恢复，设置起始epoch
        if self.config.resume_from:
            logger.info(f"Resuming from {self.config.resume_from}")
            self.accelerator.load_state(self.config.resume_from)
            self.first_epoch = int(self.config.resume_from.split("_")[-1]) + 1
        else:
            self.first_epoch = 0

    def _save_model_hook(self, models, weights, output_dir):
        assert len(models) == 1
        if isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # 确保accelerate不尝试处理模型的保存

    def _fix_seed(self):
        assert self.accelerator, "should call after init accelerator"
        # set seed (device_specific is very important to get different prompts on different devices)
        np.random.seed(self.config.seed or 114514)
        random_seeds = np.random.randint(0, 100000, size=self.available_devices)
        device_seed = random_seeds[self.accelerator.process_index]  # type: ignore
        set_seed(int(device_seed), device_specific=True)

    def _load_model_hook(self, models, input_dir):
        assert len(models) == 1
        if isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # 确保accelerate不尝试处理模型的加载

    def train(self):
        """执行训练过程"""
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {self.config.num_epochs}")
        logger.info(f"  Sample batch size per device = {self.config.sample_batch_size}")
        logger.info(f"  Train batch size per device = {self.config.train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.train_gradient_accumulation_steps}")
        logger.info("")
        logger.info(f"  Total number of samples per epoch = {self.samples_per_epoch}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_train_batch_size}"
        )
        logger.info(
            f"  Number of gradient updates per inner epoch = {self.samples_per_epoch // self.total_train_batch_size}"
        )
        logger.info(f"  Number of inner epochs = {self.config.train_num_inner_epochs}")

        global_step = 0
        for epoch in range(self.first_epoch, self.config.num_epochs):
            global_step = self.epoch_loop(epoch, global_step)

    def _sample(self, epoch, global_step):
        """采样并计算奖励"""
        samples = []
        prompts = []
        for i in t(
            range(self.config.sample_num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not self.accelerator.is_local_main_process,
            position=0,
        ):
            # 生成提示
            prompts, prompt_metadata = zip(*[self.prompt_fn() for _ in range(self.config.sample_batch_size)])

            # 编码提示
            prompt_ids = self.pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.pipeline.tokenizer.model_max_length,
            ).input_ids
            prompt_embeds = self.pipeline.text_encoder(prompt_ids)[0]

            # 采样
            with self.autocast():
                images, _, latents, log_probs = pipeline_with_logprob(
                    self.pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=self.sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=self.config.sample_guidance_scale,
                    eta=self.config.sample_eta,
                    output_type="pt",
                )

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = self.pipeline.scheduler.timesteps.repeat(
                self.config.sample_batch_size, 1
            )  # (batch_size, num_steps)

            # 直接计算奖励，不使用executor
            rewards, reward_metadata = self.reward_fn(self.vqa_pipeline, images, prompts, prompt_metadata)
            rewards = torch.as_tensor(rewards, device=self.accelerator.device)

            # eval奖励计算
            eval_rewards = None
            eval_reward_metadata = None
            if epoch % self.config.sample_eval_epoch == 0:
                eval_prompts, eval_prompt_metadata = zip(
                    *[self.prompt_fn() for _ in range(self.config.sample_eval_batch_size)]
                )

                # 编码提示
                eval_prompt_ids = self.pipeline.tokenizer(
                    eval_prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.pipeline.tokenizer.model_max_length,
                ).input_ids
                eval_prompt_embeds = self.pipeline.text_encoder(eval_prompt_ids)[0]

                # 生成与eval_batch大小匹配的负面提示嵌入
                neg_prompt_embed = self.pipeline.text_encoder(
                    self.pipeline.tokenizer(
                        [""],
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=self.pipeline.tokenizer.model_max_length,
                    ).input_ids
                )[0]
                eval_sample_neg_prompt_embeds = neg_prompt_embed.repeat(self.config.sample_eval_batch_size, 1, 1)

                # 采样
                with self.autocast():
                    eval_images, _, _, _ = pipeline_with_logprob(
                        self.pipeline,
                        prompt_embeds=eval_prompt_embeds,
                        negative_prompt_embeds=eval_sample_neg_prompt_embeds,
                        num_inference_steps=self.config.sample_num_steps,
                        guidance_scale=self.config.sample_guidance_scale,
                        eta=self.config.sample_eta,
                        output_type="pt",
                    )

                # 直接计算eval奖励
                eval_rewards_result, eval_reward_metadata = self.reward_fn(
                    self.vqa_pipeline, eval_images, eval_prompts, eval_prompt_metadata
                )
                eval_rewards = torch.as_tensor(eval_rewards_result, device=self.accelerator.device)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # 每个条目是时间步t之前的潜在变量
                    "next_latents": latents[:, 1:],  # 每个条目是时间步t之后的潜在变量
                    "log_probs": log_probs,
                    "rewards": rewards,
                    "eval_rewards": eval_rewards,
                }
            )

        # 将样本整合到字典中，其中每个条目的形状为(num_batches_per_epoch * sample.batch_size, ...)
        zip_samples = {k: torch.cat([s[k] for s in samples], dim=0) for k in samples[0].keys()}

        # 跨进程收集奖励
        rewards = self.accelerator.gather(zip_samples["rewards"]).cpu().numpy()

        # 记录奖励和图像
        if "eval_rewards" in zip_samples:
            eval_rewards = self.accelerator.gather(zip_samples["eval_rewards"]).cpu().numpy()
            self.accelerator.log(
                {
                    "reward": eval_rewards,
                    "num_samples": epoch * self.available_devices * self.config.sample_batch_size,
                    "reward_mean": eval_rewards.mean(),
                    "reward_std": eval_rewards.std(),
                },
                step=global_step,
            )
        else:
            self.accelerator.log(
                {
                    "reward": rewards,
                    "num_samples": epoch * self.available_devices * self.config.sample_batch_size,
                    "reward_mean": rewards.mean(),
                    "reward_std": rewards.std(),
                },
                step=global_step,
            )
        # 这是一个hack，强制wandb将图像记录为JPEG而不是PNG
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            self.accelerator.log(
                {
                    "images": [
                        wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt:.25} | {reward:.2f}")
                        for i, (prompt, reward) in enumerate(zip(prompts, rewards))
                    ],
                },
                step=global_step,
            )

        return zip_samples, prompts, rewards

    def epoch_loop(self, epoch, global_step):
        """执行一个完整的epoch循环，包括采样和训练"""
        #################### 采样 ####################
        self.pipeline.unet.eval()
        samples, prompts, rewards = self._sample(epoch, global_step)

        # 根据提示跟踪每个提示的均值/标准差
        if self.config.per_prompt_stat_tracking:
            # 跨进程收集提示
            prompt_ids = self.accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = self.pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = self.stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # 解除优势的收集；我们只需要保留与该进程上的样本相对应的条目
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index]
            .to(self.accelerator.device)
        )

        del samples["rewards"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert total_batch_size == self.config.sample_batch_size * self.config.sample_num_batches_per_epoch
        assert num_timesteps == self.config.sample_num_steps

        #################### 训练 ####################
        for inner_epoch in range(self.config.train_num_inner_epochs):
            # 创建数据集
            dataset = DDPODataset(samples, self.config.train_batch_size)

            # 创建DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.train_batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.dataloader_pin_memory,
            )

            # 使用accelerate准备DataLoader
            dataloader = self.accelerator.prepare(dataloader)

            # 训练
            self.pipeline.unet.train()
            info = defaultdict(list)

            for i, batch in t(
                enumerate(dataloader),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not self.accelerator.is_local_main_process,
            ):
                self.step(batch, i, epoch, inner_epoch, global_step, info)
                # 这里每个样本后递增global_step
                global_step += 1

            # 确保我们在内部epoch结束时执行了优化步骤
            assert self.accelerator.sync_gradients

        if epoch != 0 and epoch % self.config.save_freq == 0 and self.accelerator.is_main_process:
            self.accelerator.save_state()

        return global_step

    def step(self, batch, i, epoch, inner_epoch, global_step, info):
        """进行单步训练"""
        if self.config.train_cfg:
            # 将负面提示连接到样本提示以避免两次前向传递
            embeds = torch.cat([self.train_neg_prompt_embeds, batch["prompt_embeds"]])
        else:
            embeds = batch["prompt_embeds"]

        for j in t(
            range(self.num_train_timesteps),
            desc="Timestep",
            position=1,
            leave=False,
            disable=not self.accelerator.is_local_main_process,
        ):
            with self.accelerator.accumulate(self.pipeline.unet):
                with self.autocast():
                    if self.config.train_cfg:
                        noise_pred = self.pipeline.unet(
                            torch.cat([batch["latents"][:, j]] * 2),
                            torch.cat([batch["timesteps"][:, j]] * 2),
                            embeds,
                        ).sample
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.config.sample_guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                    else:
                        noise_pred = self.pipeline.unet(batch["latents"][:, j], batch["timesteps"][:, j], embeds).sample
                    # 计算给定latents的next_latents的对数概率
                    _, log_prob = ddim_step_with_logprob(
                        self.pipeline.scheduler,
                        noise_pred,
                        batch["timesteps"][:, j],
                        batch["latents"][:, j],
                        eta=self.config.sample_eta,
                        prev_sample=batch["next_latents"][:, j],
                    )

                # ppo逻辑
                advantages = torch.clamp(
                    batch["advantages"], -self.config.train_adv_clip_max, self.config.train_adv_clip_max
                )
                ratio = torch.exp(log_prob - batch["log_probs"][:, j])
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio, 1.0 - self.config.train_clip_range, 1.0 + self.config.train_clip_range
                )
                loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                # 调试值
                # John Schulman说(ratio - 1) - log(ratio)是更好的
                # 估计器，但大多数现有代码使用这个所以...
                # http://joschu.net/blog/kl-approx.html
                info["approx_kl"].append(0.5 * torch.mean((log_prob - batch["log_probs"][:, j]) ** 2))
                info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > self.config.train_clip_range).float()))
                info["loss"].append(loss)

                # 反向传播
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.trainable_layers.parameters(), self.config.train_max_grad_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

            # 检查accelerator是否在后台执行了优化步骤
            if self.accelerator.sync_gradients:
                assert (j == self.num_train_timesteps - 1) and (
                    i + 1
                ) % self.config.train_gradient_accumulation_steps == 0
                # 记录与训练相关的内容
                info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                info = self.accelerator.reduce(info, reduction="mean")
                info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                self.accelerator.log(info, step=global_step)
                info = defaultdict(list)

                # 注意：不再在这里增加global_step
