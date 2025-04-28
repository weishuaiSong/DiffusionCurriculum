from dataclasses import dataclass, field, asdict
from typing import Callable, Any, Optional
import torch
import os
import datetime
from concurrent import futures
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from train.curriculum import Curriculum
from train.trainer.common.pipeline_with_logprob import pipeline_with_logprob
import tqdm
from functools import partial

t = partial(tqdm.tqdm, dynamic_ncols=True)
logger = get_logger(__name__)


class TrainerOld(DDPOTrainer):
    def __init__(
        self,
        curriculum: Curriculum,
        update_target_difficulty: Callable[[int], None],
        config: DDPOConfig,
        reward_function: Callable[[torch.Tensor, tuple[str], tuple[Any]], torch.Tensor],
        prompt_function: Callable[[], tuple[str, Any]],
        sd_pipeline: DDPOStableDiffusionPipeline,
        image_samples_hook: Optional[Callable[[Any, Any, Any], Any]] = None,
    ):
        super().__init__(config, reward_function, prompt_function, sd_pipeline, image_samples_hook)
        self.curriculum = curriculum
        self.update_target_difficulty = update_target_difficulty

    def compute_rewards(self, prompt_image_pairs, is_async: bool = False):
        rewards = super().compute_rewards(prompt_image_pairs, is_async)
        metadata = {
            "prompt_image_pairs": prompt_image_pairs,
            "rewards": rewards,
            "current_step": self.num_train_timesteps,
        }
        target_difficulty = self.curriculum.infer_target_difficulty(metadata=metadata)
        self.update_target_difficulty(target_difficulty)
        return rewards


@dataclass
class Config:
    # 基本配置
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
    sample_save_interval: int = 100

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
    train_adv_clip_max: float = 5.0
    train_timestep_fraction: float = 1.0
    
    # DDPO特有参数
    train_clip_range: float = 1e-4
    
    # 统计跟踪相关配置
    per_prompt_stat_tracking: bool = True
    per_prompt_stat_tracking_buffer_size: int = 16
    per_prompt_stat_tracking_min_count: int = 16
    
    # 提示词和奖励函数
    prompt_fn: str = "simple_animals"
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
            self.run_name = f"ddpo_{unique_id}"
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
        
        # 创建accelerator
        self.accelerator = Accelerator(
            log_with="wandb",
            mixed_precision=self.config.mixed_precision,
            project_config=accelerator_config,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps * self.num_train_timesteps,
        )
        
        self.available_devices = self.accelerator.num_processes
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name="ddpo-pytorch", config=asdict(self.config), init_kwargs={"wandb": {"name": self.run_name}}
            )
        logger.info(f"\n{self.config}")
        
        # 设置随机种子
        if self.config.seed is not None:
            set_seed(self.config.seed)
        
        # 加载模型和调度器
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.sd_model, revision=self.config.sd_revision
        )
        if self.config.use_xformers:
            self.sd_pipeline.enable_xformers_memory_efficient_attention()
            
        # 冻结不需要训练的模型参数
        self.sd_pipeline.vae.requires_grad_(False)
        self.sd_pipeline.text_encoder.requires_grad_(False)
        self.sd_pipeline.unet.requires_grad_(False)
        
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
        
        trainable_layers = self.sd_pipeline.unet
        
        # 设置检查点保存和加载
        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)
        
        # 设置TensorFloat32（如果允许）
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            trainable_layers.parameters(),
            lr=self.config.train_learning_rate,
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
        
        # 自动混合精度设置
        self.autocast = self.accelerator.autocast
        
        # 使用accelerator准备所有组件
        trainable_layers, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)
        
        # 创建异步执行器
        self.executor = futures.ThreadPoolExecutor(max_workers=2)
        
    def _norm_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(os.getcwd(), path)
        
    def _save_model_hook(self, models, weights, output_dir):
        if self.accelerator.is_main_process:
            # 实现保存模型的逻辑
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, f"model_{i}"))

    def _load_model_hook(self, models, input_dir):
        # 实现加载模型的逻辑
        for i, model in enumerate(models):
            model.from_pretrained(os.path.join(input_dir, f"model_{i}"))
            
    def train(self):
        # 训练入口函数
        logger.info("Starting DDPO training...")
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Starting epoch {epoch}")
            # 这里实现每个epoch的训练逻辑
            # 通常包括采样、计算奖励、更新模型等步骤
            
            # 根据训练结果更新难度
            target_difficulty = self.curriculum.infer_target_difficulty({})
            self.update_target_difficulty(target_difficulty)
            
            # 这里应该实现完整的训练循环
            
            # 保存模型
            if (epoch + 1) % self.config.save_freq == 0 or epoch == self.config.num_epochs - 1:
                self.accelerator.save_state()
                
            global_step += 1
            
        return global_step
