from dataclasses import asdict, dataclass, field
from collections import defaultdict
import os
import copy
import datetime
from concurrent import futures
from train.trainer.common.pipeline_with_logprob import pipeline_with_logprob

from train.trainer.common.ddim_with_logprob import ddim_step_with_logprob
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
import numpy as np
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from typing import Any, Callable

from train.curriculum import Curriculum

t = partial(tqdm.tqdm, dynamic_ncols=True)

logger = get_logger(__name__)


@dataclass
class Config:
    sample_num_step: int = field(default=50)
    train_num_step: int = field(default=1000)
    timestep_fraction: float = field(default=0.8)
    log_dir: str = field(default="logs")
    sd_model: str = field(default="runwayml/stable-diffusion-v1-5")
    sd_revision: str = field(default="main")
    learning_rate: float = field(default=1e-4)
    eval_epoch: int = field(default=2)

    # 顶层配置
    run_name: str = ""
    seed: int = 0
    logdir: str = "logs"
    num_epochs: int = 400
    save_freq: int = 400
    num_checkpoint_limit: int = 10
    mixed_precision: str = "fp16"
    allow_tf32: bool = True
    resume_from: str = ""
    use_xformers: bool = False

    # 采样相关配置
    sample_num_steps: int = 20
    sample_eta: float = 1.0
    sample_guidance_scale: float = 5.0
    sample_batch_size: int = 1
    sample_num_batches_per_epoch: int = 2
    sample_eval_batch_size: int = 8

    # 训练相关配置
    train_batch_size: int = 1
    train_learning_rate: float = 3e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    gradient_accumulation_steps: int = 1
    train_max_grad_norm: float = 1.0
    num_inner_epochs: int = 1
    train_cfg: bool = True
    train_timestep_fraction: float = 1.0
    train_activation_checkpointing: bool = False

    # D3PO特有参数
    train_eps: float = 0.1
    train_beta: float = 1.0

    # 提示词和奖励函数
    prompt_fn: str = "simple_animals"
    prompt_fn_kwargs: dict = field(default_factory=dict)
    reward_fn: str = "jpeg_compressibility"


class Trainer:
    def __init__(
        self,
        curriculum: Curriculum,
        update_target_difficulty: Callable[[int], None],
        config: Config,
        reward_function: Callable[[torch.Tensor, tuple[str], tuple[Any]], torch.Tensor],
        prompt_function: Callable[[], tuple[str, Any]],
    ) -> None:
        self.curriculum = curriculum
        self.update_target_difficulty = update_target_difficulty
        self.config = config
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        if not self.config.run_name:
            self.run_name = f"d3po_{unique_id}"
        else:
            self.run_name = f"{self.config.run_name}_{unique_id}"

        if self.config.resume_from:
            self.config.resume_from = self._norm_path(self.config.resume_from)

        # 训练时使用的时间步数
        self.num_train_timesteps = int(self.config.sample_num_steps * self.config.train_timestep_fraction)

        accelerator_config = ProjectConfiguration(
            project_dir=os.path.join(self.config.log_dir, self.run_name),
            automatic_checkpoint_naming=True,
            total_limit=self.config.num_checkpoint_limit,
        )

        self.accelerator = Accelerator(
            log_with="wandb",
            mixed_precision=self.config.mixed_precision,
            project_config=accelerator_config,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps * self.num_train_timesteps,
        )
        self.available_devices = self.accelerator.num_processes
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name="d3po-pytorch", config=asdict(self.config), init_kwargs={"wandb": {"name": self.run_name}}
            )
        logger.info(f"\n{self.config}")

        # 设置随机种子
        self._fix_seed()

        # 加载模型和调度器
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.sd_model, revision=self.config.sd_revision, torch_dtype=torch.float16
        )
        if self.config.use_xformers:
            self.sd_pipeline.enable_xformers_memory_efficient_attention()

        # 冻结不需要训练的模型参数
        self.sd_pipeline.vae.requires_grad_(False)
        self.sd_pipeline.text_encoder.requires_grad_(False)
        self.sd_pipeline.unet.requires_grad_(True)
        if self.config.train_activation_checkpointing:
            self.sd_pipeline.unet.enable_gradient_checkpointing()

        # 禁用安全检查器
        self.sd_pipeline.safety_checker = None

        # 设置进度条
        self.sd_pipeline.set_progress_bar_config(
            position=1,
            disable=not self.accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )

        # 切换到DDIM采样器
        self.sd_pipeline.scheduler = DDIMScheduler.from_config(self.sd_pipeline.scheduler.config)

        # 混合精度设置
        inference_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16

        # 将模型移到设备上并设置数据类型
        self.sd_pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.text_encoder.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.unet.to(self.accelerator.device, dtype=inference_dtype)
        self.ref = copy.deepcopy(self.sd_pipeline.unet)
        for param in self.ref.parameters():
            param.requires_grad = False

        trainable_layers = self.sd_pipeline.unet

        # 设置检查点保存和加载
        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)

        # 设置TensorFloat32（如果允许）
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        optimizer_cls = torch.optim.AdamW

        self.optimizer = optimizer_cls(
            trainable_layers.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )

        # 准备提示词函数和奖励函数
        self.prompt_fn = prompt_function
        self.reward_fn = reward_function

        # 生成负面提示词嵌入
        neg_prompt_embed = self.sd_pipeline.text_encoder(
            self.sd_pipeline.tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
        )[0]
        self.sample_neg_prompt_embeds = neg_prompt_embed.repeat(self.config.sample_batch_size, 1, 1)
        self.train_neg_prompt_embeds = neg_prompt_embed.repeat(self.config.train_batch_size, 1, 1)

        # 自动混合精度设置
        self.autocast = self.accelerator.autocast

        # 使用accelerator准备所有组件
        trainable_layers, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)

        # 创建异步执行器
        self.executor = futures.ThreadPoolExecutor(max_workers=2)

        # 计算每个epoch的样本数和总训练批次大小
        self.samples_per_epoch = (
            self.config.sample_batch_size * self.accelerator.num_processes * self.config.sample_num_batches_per_epoch
        )
        self.total_train_batch_size = (
            self.config.train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps
        )

        # 检查批次大小配置
        assert self.config.sample_batch_size >= self.config.train_batch_size
        assert self.config.sample_batch_size % self.config.train_batch_size == 0
        assert self.samples_per_epoch % self.total_train_batch_size == 0

        # 如果从检查点恢复，加载状态
        if self.config.resume_from:
            logger.info(f"Resuming from {self.config.resume_from}")
            self.accelerator.load_state(self.config.resume_from)
            self.first_epoch = int(self.config.resume_from.split("_")[-1]) + 1
        else:
            self.first_epoch = 0

    def _fix_seed(self):
        """设置随机种子，确保每个设备使用不同的种子"""
        np.random.seed(self.config.seed)
        random_seeds = np.random.randint(0, 100000, size=self.available_devices)
        device_seed = random_seeds[self.accelerator.process_index]
        set_seed(int(device_seed), device_specific=True)

    def _norm_path(self, path: str) -> str:
        """标准化路径并自动找到最新的检查点"""
        res = os.path.normpath(os.path.expanduser(path))
        if "checkpoint_" not in os.path.basename(res):
            # 在目录中找到最新的检查点
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(path)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {path}")
            res = os.path.join(
                path,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        return res

    def _save_model_hook(self, models, weights, output_dir):
        """保存模型的钩子函数"""
        assert len(models) == 1
        if isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # 确保accelerate不会尝试处理模型的保存

    def _load_model_hook(self, models, input_dir):
        """加载模型的钩子函数"""
        assert len(models) == 1
        if isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # 确保accelerate不会尝试处理模型的加载

    def _compare(self, a, b):
        """
        支持多维比较。默认维度为1。可以添加多个奖励而不仅仅是一个来判断图像的偏好。
        例如：A: clipscore-30 blipscore-10 LAION美学评分-6.0；B: 20, 8, 5.0，则A优于B
        如果C: 40, 4, 4.0，由于C[0] = 40 > A[0]且C[1] < A[1]，我们不认为C优于A或A优于C
        """
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
        if len(a.shape) == 1:
            a = a[..., None]
            b = b[..., None]

        a_dominates = torch.logical_and(torch.all(a <= b, dim=1), torch.any(a < b, dim=1))
        b_dominates = torch.logical_and(torch.all(b <= a, dim=1), torch.any(b < a, dim=1))

        c = torch.zeros([a.shape[0], 2], dtype=torch.float, device=a.device)

        c[a_dominates] = torch.tensor([-1.0, 1.0], device=a.device)
        c[b_dominates] = torch.tensor([1.0, -1.0], device=a.device)

        return c

    def train(self):
        """训练方法"""
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

        global_step = 0
        for epoch in range(self.first_epoch, self.config.num_epochs):
            global_step = self.epoch_loop(global_step, epoch)

    def _sample(self, epoch: int, global_step: int):
        """采样方法"""
        samples = []
        prompt_metadata = None

        for i in t(
            range(self.config.sample_num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not self.accelerator.is_local_main_process,
            position=0,
        ):
            # 生成提示词
            prompts1, prompt_metadata = zip(*[self.prompt_fn() for _ in range(self.config.sample_batch_size)])
            prompts2 = prompts1

            # 编码提示词
            prompt_ids1 = self.sd_pipeline.tokenizer(
                prompts1,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)

            prompt_ids2 = self.sd_pipeline.tokenizer(
                prompts2,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
            prompt_embeds1 = self.sd_pipeline.text_encoder(prompt_ids1)[0]
            prompt_embeds2 = self.sd_pipeline.text_encoder(prompt_ids2)[0]

            # 采样
            with self.autocast():
                images1, _, latents1, log_probs1 = pipeline_with_logprob(
                    self.sd_pipeline,
                    prompt_embeds=prompt_embeds1,
                    negative_prompt_embeds=self.sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=self.config.sample_guidance_scale,
                    eta=self.config.sample_eta,
                    output_type="pt",
                )
                latents1 = torch.stack(latents1, dim=1)
                log_probs1 = torch.stack(log_probs1, dim=1)
                images2, _, latents2, log_probs2 = pipeline_with_logprob(
                    self.sd_pipeline,
                    prompt_embeds=prompt_embeds2,
                    negative_prompt_embeds=self.sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=self.config.sample_guidance_scale,
                    eta=self.config.sample_eta,
                    output_type="pt",
                    latents=latents1[:, 0, :, :, :],
                )
                latents2 = torch.stack(latents2, dim=1)
                log_probs2 = torch.stack(log_probs2, dim=1)

            latents = torch.stack([latents1, latents2], dim=1)  # (batch_size, 2, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack([log_probs1, log_probs2], dim=1)  # (batch_size, num_steps, 1)
            prompt_embeds = torch.stack([prompt_embeds1, prompt_embeds2], dim=1)
            images = torch.stack([images1, images2], dim=1)
            current_latents = latents[:, :, :-1]
            next_latents = latents[:, :, 1:]
            timesteps = self.sd_pipeline.scheduler.timesteps.repeat(
                self.config.sample_batch_size, 1
            )  # (batch_size, num_steps)

            # 异步计算奖励
            rewards1 = self.executor.submit(self.reward_fn, images1, prompts1, prompt_metadata).result()[0]
            # 让步以确保奖励计算开始
            rewards2 = self.executor.submit(self.reward_fn, images2, prompts2, prompt_metadata).result()[0]
            if isinstance(rewards1, np.ndarray):
                rewards = np.c_[rewards1, rewards2]
            else:
                rewards1 = rewards1.cpu().detach().numpy()
                rewards2 = rewards2.cpu().detach().numpy()
                rewards = np.c_[rewards1, rewards2]

            eval_rewards = None
            if epoch % self.config.eval_epoch == 0:
                eval_prompts, eval_prompt_metadata = zip(
                    *[self.prompt_fn() for _ in range(self.config.sample_eval_batch_size)]
                )

                # 编码提示词
                eval_prompt_ids = self.sd_pipeline.tokenizer(
                    eval_prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.sd_pipeline.tokenizer.model_max_length,
                ).input_ids.to(self.accelerator.device)
                eval_prompt_embeds = self.sd_pipeline.text_encoder(eval_prompt_ids)[0]
                eval_sample_neg_prompt_embeds = self.sample_neg_prompt_embeds.repeat(
                    self.config.sample_eval_batch_size // self.config.sample_batch_size, 1, 1
                )

                # 采样
                with self.autocast():
                    eval_images, _, _, _ = pipeline_with_logprob(
                        self.sd_pipeline,
                        prompt_embeds=eval_prompt_embeds,
                        negative_prompt_embeds=eval_sample_neg_prompt_embeds,
                        num_inference_steps=self.config.sample_num_steps,
                        guidance_scale=self.config.sample_guidance_scale,
                        eta=self.config.sample_eta,
                        output_type="pt",
                    )
                eval_rewards = self.executor.submit(
                    self.reward_fn, eval_images, eval_prompts, eval_prompt_metadata
                ).result()[0]
                samples.append(
                    {
                        "prompt_embeds": prompt_embeds,
                        "prompts": prompts1,
                        "timesteps": timesteps,
                        "latents": current_latents,  # 每个条目是时间步t之前的潜变量
                        "next_latents": next_latents,  # 每个条目是时间步t之后的潜变量
                        "log_probs": log_probs,
                        "images": images,
                        "rewards": torch.as_tensor(rewards, device=self.accelerator.device),
                        "eval_rewards": torch.as_tensor(eval_rewards, device=self.accelerator.device),
                    }
                )
            else:
                prompts1 = list(prompts1)
                samples.append(
                    {
                        "prompt_embeds": prompt_embeds,
                        "prompts": prompts1,
                        "timesteps": timesteps,
                        "latents": current_latents,
                        "next_latents": next_latents,
                        "log_probs": log_probs,
                        "images": images,
                        "rewards": torch.as_tensor(rewards, device=self.accelerator.device),
                    }
                )

        prompts = samples[0]["prompts"]
        del samples[0]["prompts"]
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        images = samples["images"]
        rewards = self.accelerator.gather(samples["rewards"]).cpu().numpy()

        # 记录基于奖励模型的更好图像奖励
        if "eval_rewards" in samples:
            eval_rewards = self.accelerator.gather(samples["eval_rewards"]).cpu().numpy()
            self.accelerator.log(
                {
                    "eval_reward": eval_rewards,
                    "num_samples": epoch * self.available_devices * self.config.sample_batch_size,
                    "eval_reward_mean": eval_rewards.mean(),
                    "eval_reward_std": eval_rewards.std(),
                },
                step=global_step,
            )
            del samples["eval_rewards"]
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
                pil = Image.fromarray((image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            self.accelerator.log(
                {
                    "images": [
                        wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt:.25} | {reward:.2f}")
                        for i, (prompt, reward) in enumerate(zip(prompts, rewards[:, 0]))
                    ],
                },
                step=global_step,
            )

        # 保存提示词
        del samples["images"]
        torch.cuda.empty_cache()

        return samples, prompts, rewards

    def epoch_loop(self, global_step: int, epoch: int):
        """一个完整训练周期的循环"""
        #################### 采样 ####################
        self.sd_pipeline.unet.eval()

        samples, prompts, rewards = self._sample(epoch, global_step)

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert total_batch_size == self.config.sample_batch_size * self.config.sample_num_batches_per_epoch
        assert num_timesteps == self.config.sample_num_steps
        orig_sample = copy.deepcopy(samples)

        #################### 训练 ####################
        for inner_epoch in range(self.config.num_inner_epochs):
            # 沿批次维度随机打乱样本
            perm = torch.randperm(total_batch_size, device=self.accelerator.device)
            samples = {k: v[perm] for k, v in orig_sample.items()}

            # 对每个样本沿时间维度独立随机打乱
            perms = torch.stack(
                [torch.randperm(num_timesteps, device=self.accelerator.device) for _ in range(total_batch_size)]
            )
            for key in ["latents", "next_latents"]:
                tmp = samples[key].permute(0, 2, 3, 4, 5, 1)[
                    torch.arange(total_batch_size, device=self.accelerator.device)[:, None], perms
                ]
                samples[key] = tmp.permute(0, 5, 1, 2, 3, 4)
            samples["timesteps"] = (
                samples["timesteps"][torch.arange(total_batch_size, device=self.accelerator.device)[:, None], perms]
                .unsqueeze(1)
                .repeat(1, 2, 1)
            )
            tmp = samples["log_probs"].permute(0, 2, 1)[
                torch.arange(total_batch_size, device=self.accelerator.device)[:, None], perms
            ]
            samples["log_probs"] = tmp.permute(0, 2, 1)

            # 重新分批用于训练
            samples_batched = {k: v.reshape(-1, self.config.train_batch_size, *v.shape[1:]) for k, v in samples.items()}
            # 字典列表 -> 列表字典，便于迭代
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            # 训练
            self.sd_pipeline.unet.train()
            info = defaultdict(list)
            for i in t(
                range(0, total_batch_size, self.config.train_batch_size),
                desc="更新",
                position=2,
                leave=False,
            ):
                self.step(samples, i, epoch, inner_epoch, global_step, info)
                global_step += 1

            # 确保在内部epoch结束时进行了优化步骤
            assert self.accelerator.sync_gradients

        if epoch != 0 and epoch % self.config.save_freq == 0 and self.accelerator.is_main_process:
            self.accelerator.save_state()

        return global_step

    def step(self, samples, step, epoch, inner_epoch, global_step, info):
        """执行一步训练"""
        sample_0 = {}
        sample_1 = {}
        for key, value in samples.items():
            sample_0[key] = value[step : step + self.config.train_batch_size, 0]
            sample_1[key] = value[step : step + self.config.train_batch_size, 1]

        if self.config.train_cfg:
            # 将负面提示词与样本提示词连接，避免两次前向传递
            embeds_0 = torch.cat([self.train_neg_prompt_embeds, sample_0["prompt_embeds"]])
            embeds_1 = torch.cat([self.train_neg_prompt_embeds, sample_1["prompt_embeds"]])
        else:
            embeds_0 = sample_0["prompt_embeds"]
            embeds_1 = sample_1["prompt_embeds"]

        for j in t(
            range(self.num_train_timesteps),
            desc="时间步",
            position=3,
            leave=False,
            disable=not self.accelerator.is_local_main_process,
        ):
            with self.accelerator.accumulate(self.sd_pipeline.unet):
                with self.autocast():
                    if self.config.train_cfg:
                        noise_pred_0 = self.sd_pipeline.unet(
                            torch.cat([sample_0["latents"][:, j]] * 2),
                            torch.cat([sample_0["timesteps"][:, j]] * 2),
                            embeds_0,
                        ).sample
                        noise_pred_uncond_0, noise_pred_text_0 = noise_pred_0.chunk(2)
                        noise_pred_0 = noise_pred_uncond_0 + self.config.sample_guidance_scale * (
                            noise_pred_text_0 - noise_pred_uncond_0
                        )

                        noise_ref_pred_0 = self.ref(
                            torch.cat([sample_0["latents"][:, j]] * 2),
                            torch.cat([sample_0["timesteps"][:, j]] * 2),
                            embeds_0,
                        ).sample
                        noise_ref_pred_uncond_0, noise_ref_pred_text_0 = noise_ref_pred_0.chunk(2)
                        noise_ref_pred_0 = noise_ref_pred_uncond_0 + self.config.sample_guidance_scale * (
                            noise_ref_pred_text_0 - noise_ref_pred_uncond_0
                        )

                        noise_pred_1 = self.sd_pipeline.unet(
                            torch.cat([sample_1["latents"][:, j]] * 2),
                            torch.cat([sample_1["timesteps"][:, j]] * 2),
                            embeds_1,
                        ).sample
                        noise_pred_uncond_1, noise_pred_text_1 = noise_pred_1.chunk(2)
                        noise_pred_1 = noise_pred_uncond_1 + self.config.sample_guidance_scale * (
                            noise_pred_text_1 - noise_pred_uncond_1
                        )

                        noise_ref_pred_1 = self.ref(
                            torch.cat([sample_1["latents"][:, j]] * 2),
                            torch.cat([sample_1["timesteps"][:, j]] * 2),
                            embeds_1,
                        ).sample
                        noise_ref_pred_uncond_1, noise_ref_pred_text_1 = noise_ref_pred_1.chunk(2)
                        noise_ref_pred_1 = noise_ref_pred_uncond_1 + self.config.sample_guidance_scale * (
                            noise_ref_pred_text_1 - noise_ref_pred_uncond_1
                        )

                    else:
                        noise_pred_0 = self.sd_pipeline.unet(
                            sample_0["latents"][:, j], sample_0["timesteps"][:, j], embeds_0
                        ).sample
                        noise_ref_pred_0 = self.ref(
                            sample_0["latents"][:, j], sample_0["timesteps"][:, j], embeds_0
                        ).sample

                        noise_pred_1 = self.sd_pipeline.unet(
                            sample_1["latents"][:, j], sample_1["timesteps"][:, j], embeds_1
                        ).sample
                        noise_ref_pred_1 = self.ref(
                            sample_1["latents"][:, j], sample_1["timesteps"][:, j], embeds_1
                        ).sample

                    # 计算当前模型下next_latents相对于latents的对数概率
                    _, total_prob_0 = ddim_step_with_logprob(
                        self.sd_pipeline.scheduler,
                        noise_pred_0,
                        sample_0["timesteps"][:, j],
                        sample_0["latents"][:, j],
                        eta=self.config.sample_eta,
                        prev_sample=sample_0["next_latents"][:, j],
                    )
                    _, total_ref_prob_0 = ddim_step_with_logprob(
                        self.sd_pipeline.scheduler,
                        noise_ref_pred_0,
                        sample_0["timesteps"][:, j],
                        sample_0["latents"][:, j],
                        eta=self.config.sample_eta,
                        prev_sample=sample_0["next_latents"][:, j],
                    )
                    _, total_prob_1 = ddim_step_with_logprob(
                        self.sd_pipeline.scheduler,
                        noise_pred_1,
                        sample_1["timesteps"][:, j],
                        sample_1["latents"][:, j],
                        eta=self.config.sample_eta,
                        prev_sample=sample_1["next_latents"][:, j],
                    )
                    _, total_ref_prob_1 = ddim_step_with_logprob(
                        self.sd_pipeline.scheduler,
                        noise_ref_pred_1,
                        sample_1["timesteps"][:, j],
                        sample_1["latents"][:, j],
                        eta=self.config.sample_eta,
                        prev_sample=sample_1["next_latents"][:, j],
                    )

                # 人类偏好比较
                human_prefer = self._compare(sample_0["rewards"], sample_1["rewards"])

                # 裁剪Q值
                ratio_0 = torch.clamp(
                    torch.exp(total_prob_0 - total_ref_prob_0), 1 - self.config.train_eps, 1 + self.config.train_eps
                )
                ratio_1 = torch.clamp(
                    torch.exp(total_prob_1 - total_ref_prob_1), 1 - self.config.train_eps, 1 + self.config.train_eps
                )

                # D3PO损失函数
                loss = -torch.log(
                    torch.sigmoid(
                        self.config.train_beta * (torch.log(ratio_0)) * human_prefer[:, 0]
                        + self.config.train_beta * (torch.log(ratio_1)) * human_prefer[:, 1]
                    )
                ).mean()

                # 反向传播
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.sd_pipeline.unet.parameters(), self.config.train_max_grad_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

            # 检查加速器是否在后台执行了优化步骤
            if self.accelerator.sync_gradients:
                assert (j == self.num_train_timesteps - 1) and (step + 1) % self.config.gradient_accumulation_steps == 0
                # 记录训练相关信息
                info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                info = self.accelerator.reduce(info, reduction="mean")
                info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                self.accelerator.log(info, step=global_step)
                global_step += 1
                info = defaultdict(list)

                # 确保在内部epoch结束时进行了优化步骤
                assert self.accelerator.sync_gradients
