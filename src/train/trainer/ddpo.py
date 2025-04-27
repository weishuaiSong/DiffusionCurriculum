from dataclasses import dataclass
from typing import Callable, Any, Optional
import torch
from train.curriculum import Curriculum
from trl.trainer import DDPOConfig, DDPOTrainer


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
class Config: ...


class Trainer: ...
