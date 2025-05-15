import datetime
import os
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, gather, gather_object, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainerCallback
from transformers.utils import is_peft_available
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_rich_available
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.utils import (
    print_prompt_completions_sample,
)

from train.curriculum import Curriculum
from train.trainer.common.state_tracker import PerPromptStatTracker

if is_peft_available():
    from peft import PeftConfig
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

logger = get_logger(__name__)


@dataclass
class Config:
    # 模型配置
    pretrained_model: str = field(default="google/llava-1.6-34b-diffusion")
    pretrained_revision: str = field(default="main")
    use_lora: bool = field(default=False)

    # 随机种子
    seed: int = field(default=42)

    # 日志和检查点配置
    logdir: str = field(default="logs")
    run_name: str = field(default="")
    num_checkpoint_limit: int = field(default=10)
    save_freq: int = field(default=1)

    # 训练配置
    num_epochs: int = field(default=1)
    mixed_precision: str = field(default="bf16")
    allow_tf32: bool = field(default=True)
    resume_from: str = field(default="")

    # 采样配置
    sample_batch_size: int = field(default=4)
    sample_num_batches_per_epoch: int = field(default=4)

    # 训练配置
    train_batch_size: int = field(default=1)
    train_learning_rate: float = field(default=1e-5)
    train_num_inner_epochs: int = field(default=1)
    train_gradient_accumulation_steps: int = field(default=1)
    train_max_grad_norm: float = field(default=1.0)
    train_adv_clip_max: float = field(default=5.0)

    # 扩散模型特定配置
    diffusion_steps: int = field(default=50)
    block_length: int = field(default=128)
    max_completion_length: int = field(default=128)
    temperature: float = field(default=0.0)
    cfg_scale: float = field(default=0.0)
    remasking: str = field(default="low_confidence")
    generation_batch_size: int = field(default=8)
    mask_id: int = field(default=126336)
    p_mask_prompt: float = field(default=0.5)
    random_masking: bool = field(default=True)

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
    prompt_fn: str = field(default="simple_topics")
    prompt_fn_kwargs: dict = field(default_factory=dict)
    reward_fn: str = field(default="fluency")

    # 数据加载配置
    num_workers: int = field(default=4)
    dataloader_pin_memory: bool = field(default=True)

    def to_grpo_config(self) -> GRPOConfig:
        """
        将自定义Config转换为GRPOConfig。

        Returns:
            GRPOConfig: 转换后的GRPO配置对象
        """
        # 定义配置映射关系
        config_mapping = {
            # 扩散模型配置
            "mask_id": self.mask_id,
            "p_mask_prompt": self.p_mask_prompt,
            "random_masking": self.random_masking,
            "diffusion_steps": self.diffusion_steps,
            "block_length": self.block_length,
            "max_completion_length": self.max_completion_length,
            "cfg_scale": self.cfg_scale,
            "temperature": self.temperature,
            "remasking": self.remasking,
            "generation_batch_size": self.generation_batch_size,
            # 训练配置
            "output_dir": os.path.join(self.logdir, self.run_name),
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.train_batch_size,
            "gradient_accumulation_steps": self.train_gradient_accumulation_steps,
            "max_grad_norm": self.train_max_grad_norm,
            "learning_rate": self.train_learning_rate,
            # 优化器配置
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "weight_decay": self.adam_weight_decay,
            # 其他配置
            "num_iterations": 1,  # GRPO默认值
        }

        return GRPOConfig(**config_mapping)


class DiffuGRPOTrainer(GRPOTrainer):
    """
    Group Relative Policy Optimization (GRPO) Trainer for Diffusion Language Models.

    This class extends the GRPOTrainer to adapt it for masked diffusion language models,
    implementing efficient policy gradient estimation through conditional probabilities
    with masked tokens.

    Key features:
    - Random masking for improved robustness in multiple policy optimization updates
    - Efficient computation of per-token log probabilities for diffusion models
    - Specialized generation process for diffusion models with iterative denoising
    - Curriculum learning based on reward feedback
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        curriculum: Optional[Curriculum] = None,
        update_target_difficulty: Optional[Callable[[int], None]] = None,
        config: Optional[Config] = None,
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Store curriculum and related parameters
        self.curriculum = curriculum
        self.update_target_difficulty = update_target_difficulty
        self.last_difficulty = 0
        self.config = config or Config()

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

        # 使用新的转换函数将Config转换为GRPOConfig
        if args is None:
            args = self.config.to_grpo_config()

        # Create accelerator for training
        accelerator_config = ProjectConfiguration(
            project_dir=os.path.join(self.config.logdir, self.config.run_name),
            automatic_checkpoint_naming=True,
            total_limit=self.config.num_checkpoint_limit,
        )

        self.accelerator = Accelerator(
            log_with="wandb",
            mixed_precision=self.config.mixed_precision,
            project_config=accelerator_config,
            gradient_accumulation_steps=self.config.train_gradient_accumulation_steps,
        )

        self.available_devices = self.accelerator.num_processes
        self._fix_seed()

        # 初始化统计跟踪器
        self.stat_tracker = None
        if self.config.per_prompt_stat_tracking:
            self.stat_tracker = PerPromptStatTracker(
                self.config.per_prompt_stat_tracking_buffer_size,
                self.config.per_prompt_stat_tracking_min_count,
            )

        # 计算每个epoch的样本数和批次大小
        self.samples_per_epoch = (
            self.config.sample_batch_size * self.accelerator.num_processes * self.config.sample_num_batches_per_epoch
        )
        self.total_train_batch_size = (
            self.config.train_batch_size
            * self.accelerator.num_processes
            * self.config.train_gradient_accumulation_steps
        )

        # Configure the model for generation with appropriate parameters
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name="diffugrpo-pytorch",
                config=asdict(self.config),
                init_kwargs={"wandb": {"name": self.config.run_name}},
            )
        logger.info(f"\n{self.config}")

        # 如果从检查点恢复，设置起始epoch
        if self.config.resume_from:
            logger.info(f"Resuming from {self.config.resume_from}")
            self.accelerator.load_state(self.config.resume_from)
            self.first_epoch = int(self.config.resume_from.split("_")[-1]) + 1
        else:
            self.first_epoch = 0

        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

    def _fix_seed(self):
        """设置随机种子，确保不同进程有不同的随机种子"""
        assert self.accelerator, "should call after init accelerator"
        # set seed (device_specific is very important to get different prompts on different devices)
        np.random.seed(self.config.seed or 114514)
        random_seeds = np.random.randint(0, 100000, size=self.available_devices)
        device_seed = random_seeds[self.accelerator.process_index]  # type: ignore
        set_seed(int(device_seed), device_specific=True)

    def _save_model_hook(self, models, weights, output_dir):
        """保存模型时的钩子函数，处理模型状态的保存"""
        assert len(models) == 1
        model = self.accelerator.unwrap_model(models[0])

        # 保存模型到指定目录
        model.save_pretrained(os.path.join(output_dir, "model"))

        # 同时保存分词器
        if hasattr(self, "processing_class") and self.processing_class is not None:
            self.processing_class.save_pretrained(os.path.join(output_dir, "tokenizer"))

        # 清空权重，以便accelerate不会尝试再次处理保存
        weights.pop()

    def _load_model_hook(self, models, input_dir):
        """加载模型时的钩子函数，处理模型状态的加载"""
        assert len(models) == 1

        # 从指定目录加载模型
        model_path = os.path.join(input_dir, "model")
        if os.path.exists(model_path):
            from_model = type(models[0]).from_pretrained(model_path)
            models[0].load_state_dict(from_model.state_dict())
            del from_model

        # 同时加载分词器
        tokenizer_path = os.path.join(input_dir, "tokenizer")
        if hasattr(self, "processing_class") and self.processing_class is not None and os.path.exists(tokenizer_path):
            self.processing_class = type(self.processing_class).from_pretrained(tokenizer_path)

        # 清空模型，以便accelerate不会尝试再次处理加载
        models.pop()

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

            if epoch != 0 and epoch % self.config.save_freq == 0:
                print("Start saving...")
                self.accelerator.wait_for_everyone()
                self.accelerator.save_state()
                print("Save finished!")

    def epoch_loop(self, epoch, global_step):
        """执行一个完整的epoch循环，包括采样和训练"""
        #################### 采样 ####################
        # 将模型设置为评估模式以进行采样
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.eval()

        # 创建一个空字典列表来存储输入
        inputs_list = []

        # 为每个批次生成提示，总计self.config.sample_num_batches_per_epoch * self.config.sample_batch_size个
        for batch_idx in range(self.config.sample_num_batches_per_epoch):
            for sample_idx in range(self.config.sample_batch_size):
                # 创建一个简单的示例输入
                inputs_list.append({"prompt": [{"role": "user", "content": f"Sample {batch_idx}-{sample_idx}"}]})

        # 将输入分成批次进行处理
        batched_inputs = [
            inputs_list[i : i + self.config.train_batch_size]
            for i in range(0, len(inputs_list), self.config.train_batch_size)
        ]

        samples = []

        # 对每个批次进行处理
        for batch in batched_inputs:
            # 生成并评分完成内容
            batch_samples = self._generate_and_score_completions(batch)
            samples.append(batch_samples)

        # 将所有样本整合到一个字典中
        combined_samples = {}
        for k in samples[0].keys():
            if isinstance(samples[0][k], torch.Tensor):
                combined_samples[k] = torch.cat([s[k] for s in samples], dim=0)
            elif isinstance(samples[0][k], list):
                combined_samples[k] = [item for s in samples for item in s[k]]
            else:
                combined_samples[k] = samples[0][k]  # 假设所有批次此字段相同（如mask_seeds）

        # 释放不需要的内存
        del samples
        torch.cuda.empty_cache()

        #################### 训练 ####################
        for inner_epoch in range(self.config.train_num_inner_epochs):
            # 将模型设置为训练模式
            unwrapped_model.train()
            info = defaultdict(list)

            # 重新批处理样本以进行训练
            train_batch_size = self.config.train_batch_size

            # 将样本重新组织成训练批次
            samples_batched = {}
            for k, v in combined_samples.items():
                if isinstance(v, torch.Tensor):
                    # 重新排列为(num_batches, batch_size, ...)的形状
                    samples_batched[k] = v.reshape(-1, train_batch_size, *v.shape[1:])
                elif isinstance(v, list) and len(v) > 0:
                    # 分割列表
                    samples_batched[k] = [v[i : i + train_batch_size] for i in range(0, len(v), train_batch_size)]

            # 将字典列表转换为列表字典以便于迭代
            num_batches = len(
                next(iter([v for v in samples_batched.values() if isinstance(v, list) or isinstance(v, torch.Tensor)]))
            )
            batches = []
            for i in range(num_batches):
                batch = {}
                for k, v in samples_batched.items():
                    if isinstance(v, list):
                        batch[k] = v[i] if i < len(v) else []
                    elif isinstance(v, torch.Tensor):
                        batch[k] = v[i]
                    else:
                        batch[k] = v  # 假设是非列表/张量值
                batches.append(batch)

            # 训练循环
            for batch_idx, batch in enumerate(batches):
                # 使用现有的compute_loss方法训练模型
                loss = self.compute_loss(unwrapped_model, batch)

                # 在每个梯度累积步骤后执行优化
                self.accelerator.backward(loss)

                if (batch_idx + 1) % self.config.train_gradient_accumulation_steps == 0:
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.train_max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # 记录损失和其他指标
                info["loss"].append(loss.detach().float())

                # 更新全局步骤
                global_step += 1

                # 每个梯度更新后记录训练指标
                if (batch_idx + 1) % self.config.train_gradient_accumulation_steps == 0:
                    avg_loss = torch.mean(torch.stack(info["loss"]))
                    self.accelerator.log(
                        {
                            "train/loss": avg_loss.item(),
                            "epoch": epoch,
                            "inner_epoch": inner_epoch,
                            "step": global_step,
                        },
                        step=global_step,
                    )
                    info = defaultdict(list)

        return global_step

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.0
        cfg_scale = self.args.cfg_scale

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            prompt_completion_ids_all = []
            # Process in batches
            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx]
                batch_prompt_mask = prompt_mask[i:end_idx]
                batch_prompt_completion_ids = self.generate(
                    model=unwrapped_model,
                    prompt=batch_prompt_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=self.args.remasking,
                    mask_id=self.args.mask_id,
                )
                prompt_completion_ids_all.append(batch_prompt_completion_ids)

                del batch_prompt_ids, batch_prompt_mask, batch_prompt_completion_ids
                torch.cuda.empty_cache()

            prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        if self.args.random_masking:
            # use random seeds for every iterations in GRPO iterations
            mask_seeds = torch.randint(0, 2**12, (self.num_iterations,), device=device)
        else:
            # use fixed seeds for every iterations in GRPO iterations
            mask_seeds = [42] * self.num_iterations

        all_old_per_token_logps = []
        all_ref_per_token_logps = []
        with torch.no_grad():
            if self.num_iterations > 1:
                # repeat prompt completion ids self.num_iterations times
                prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(self.num_iterations, -1, -1)
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds
                )
                all_old_per_token_logps = old_per_token_logps
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds
                    )
                    all_ref_per_token_logps = ref_per_token_logps

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):

                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                # Convert None values to NaN
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # 更新课程学习的难度级别，如果启用了课程学习
        if self.curriculum is not None and self.update_target_difficulty is not None:
            self.last_difficulty = self.curriculum.infer_target_difficulty(
                {
                    "current_step": self._step,
                    "difficulty": self.last_difficulty,
                    "reward": rewards.mean().cpu().numpy(),
                }
            )
            self.update_target_difficulty(self.last_difficulty)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

        # 根据提示跟踪每个提示的均值/标准差
        if getattr(self, "stat_tracker", None) is not None and self.config.per_prompt_stat_tracking:
            # 跨进程收集提示
            prompt_ids_gathered = self.accelerator.gather(prompt_ids).cpu().numpy()
            prompts_text_gathered = self.processing_class.batch_decode(prompt_ids_gathered, skip_special_tokens=True)
            rewards_gathered = self.accelerator.gather(rewards).cpu().numpy()

            advantages = self.stat_tracker.update(prompts_text_gathered, rewards_gathered)

            # 解除优势的收集；我们只需要保留与该进程上的样本相对应的条目
            advantages = (
                torch.as_tensor(advantages)
                .reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index]
                .to(self.accelerator.device)
            )
        else:
            advantages = rewards - mean_grouped_rewards
            if std_grouped_rewards.abs().sum() > 0:  # 避免除以零
                advantages = advantages / (std_grouped_rewards + 1e-8)

        # Count prompts with zero std deviation
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()  # Using a small threshold
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # 记录当前难度等级，如果使用课程学习
        if self.curriculum is not None:
            self._metrics[mode]["difficulty"].append(self.last_difficulty)

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": all_old_per_token_logps,
            "ref_per_token_logps": all_ref_per_token_logps,
            "advantages": advantages,
            "mask_seeds": mask_seeds,  # Store all mask seeds for consistent mask patterns
        }

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        mask_seeds = inputs["mask_seeds"]

        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)  # only compute logits for completion tokens

        # Get the current iteration index and corresponding mask seed
        this_itr_idx = self._step % self.args.num_iterations
        this_itr_mask_seed = mask_seeds[this_itr_idx]
        input_ids = input_ids.unsqueeze(0)
        per_token_logps = self._get_per_token_logps(model, input_ids, logits_to_keep, [this_itr_mask_seed])
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][this_itr_idx].squeeze(0)
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        old_per_token_logps = (
            inputs["old_per_token_logps"][this_itr_idx].squeeze(0)
            if self.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        return loss

    def add_gumbel_noise(self, logits, temperature, dtype):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if temperature == 0.0:
            return logits  # Skip noise when temperature is 0
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def generate(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
    ):
        """generation code adopted from llada (https://github.com/ML-GSAI/LLaDA)"""
        with torch.cuda.amp.autocast(enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, : prompt.shape[1]] = prompt.clone()

            prompt_index = x != mask_id

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            # Adjust steps if needed
            steps_per_block = max(1, steps // num_blocks)

            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length

                block_mask_index = x[:, start_idx:end_idx] == mask_id
                num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps_per_block)

                for i in range(steps_per_block):
                    torch.cuda.empty_cache()
                    mask_index = x == mask_id

                    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                        with torch.cuda.amp.autocast(enabled=self.args.fp16):
                            # Handle classifier-free guidance more efficiently
                            if cfg_scale > 0.0:
                                un_x = x.clone()
                                un_x[prompt_index] = mask_id
                                x_ = torch.cat([x, un_x], dim=0)

                                # Get logits in a single forward pass
                                logits = model(x_).logits
                                logits, un_logits = torch.chunk(logits, 2, dim=0)
                                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                            else:
                                logits = model(x).logits

                            # Apply Gumbel noise for sampling
                            logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature, dtype=dtype)
                            x0 = torch.argmax(logits_with_noise, dim=-1)
                            del logits_with_noise

                            # Handle remasking strategy
                            if remasking == "low_confidence":
                                p = F.softmax(logits.to(dtype), dim=-1)
                                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                            elif remasking == "random":
                                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                            else:
                                raise NotImplementedError(remasking)

                            # Ensure we don't process tokens beyond the current block
                            x0_p[:, end_idx:] = -np.inf

                            # Update masked tokens
                            x0 = torch.where(mask_index, x0, x)
                            confidence = torch.where(mask_index, x0_p, -np.inf)

                            # Select tokens to transfer based on confidence
                            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                            for j in range(confidence.shape[0]):
                                num_tokens = num_transfer_tokens[j, i].item()
                                if num_tokens > 0:
                                    _, select_index = torch.topk(confidence[j], k=num_tokens)
                                    transfer_index[j, select_index] = True

                            x[transfer_index] = x0[transfer_index]
                            del x0, confidence, transfer_index

            return x

    def forward_process(self, batch, prompt_index, mask_id, seed=None):
        set_seed(seed)
        b, l = batch.shape
        t_p = torch.ones(b, device=batch.device) * self.args.p_mask_prompt

        # Create a random matrix to decide whether each prompt token is masked
        random_matrix = torch.rand((b, l), device=batch.device)

        # For prompt tokens: mask if random_matrix < t_p
        # For completion tokens: always mask
        is_mask_prompt = prompt_index & (random_matrix < t_p.unsqueeze(1))
        is_mask_completion = ~prompt_index  # all completion tokens are masked
        is_mask = is_mask_prompt | is_mask_completion

        # Create a noisy (masked) batch
        noisy_batch = torch.where(is_mask, mask_id, batch)

        # Build p_mask, the probability that each token is masked under this scheme
        #   - p_mask[i, j] = t_p[i] if it's a prompt token
        #   - p_mask[i, j] = 1      if it's a completion token
        p_mask = torch.where(
            prompt_index,
            t_p.unsqueeze(1),  # prompt token probability
            torch.ones_like(t_p).unsqueeze(1),  # completion token probability
        )

        return noisy_batch, p_mask

    def get_logits(self, model, batch, prompt_index, cfg_scale, mask_id):
        if cfg_scale > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = mask_id
            batch = torch.cat([batch, un_batch])

        input = batch
        logits = model(input).logits

        if cfg_scale > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        return logits

    def get_num_transfer_tokens(self, mask_index, steps):
        """
        Precompute the number of tokens to transition at each step.
        Optimized to be more efficient.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        # Create tensor once and modify in-place
        num_transfer_tokens = base.expand(-1, steps).clone()

        # Handle remainder more efficiently
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1

        return num_transfer_tokens.to(torch.int64)

    def _get_per_token_logps(self, model, input_ids, logits_to_keep, mask_seeds):
        """
        Calculate per-token log probabilities.
        """
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device
        per_token_logps = torch.zeros(num_iterations, batch_size, logits_to_keep, device=device)

        # Verify mask_seeds length: one seed per iteration
        assert (
            len(mask_seeds) == num_iterations
        ), f"Expected mask_seeds length to be {num_iterations}, got {len(mask_seeds)}"

        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True  # Mark prompt tokens as True

        # applying masks
        all_perturbed_seqs = []
        all_expanded_inputs = []
        for iter_idx, mask_seed in enumerate(mask_seeds):
            expanded_input = input_ids[iter_idx]  # [batch_size, seq_len]
            perturbed_seq, _ = self.forward_process(expanded_input, prompt_index, self.args.mask_id, seed=mask_seed)
            all_perturbed_seqs.append(perturbed_seq)
            all_expanded_inputs.append(expanded_input)

        # Concatenate all iterations into a single batch
        perturbed_seq = torch.cat(all_perturbed_seqs, dim=0)  # [num_iterations * batch_size, seq_len]
        expanded_input = torch.cat(all_expanded_inputs, dim=0)  # [num_iterations * batch_size, seq_len]

        # Get model predictions for the combined batch
        logits = self.get_logits(
            model, perturbed_seq, prompt_index, self.args.cfg_scale, self.args.mask_id
        )  # [num_iterations * batch_size, seq_len, vocab_size]

        # Calculate cross-entropy loss for completion tokens only
        completion_logits = logits[:, -logits_to_keep:, :]  # [num_iterations * batch_size, logits_to_keep, vocab_size]
        completion_targets = expanded_input[:, -logits_to_keep:]  # [num_iterations * batch_size, logits_to_keep]
        flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))
        flat_targets = completion_targets.reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        # Convert to log probabilities and reshape
        completion_log_probs = -loss.view(num_iterations * batch_size, logits_to_keep)
        per_token_logps = completion_log_probs.view(num_iterations, batch_size, logits_to_keep)

        # Clean up memory
        del perturbed_seq, logits, all_perturbed_seqs, all_expanded_inputs
        torch.cuda.empty_cache()
        per_token_logps = per_token_logps.to(torch.float32)
        return per_token_logps

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs
