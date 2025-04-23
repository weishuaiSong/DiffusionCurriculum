from dataclasses import dataclass, field
from train.ordered_dataloader import CurriculumPromptLoader
from train.scorer import VQAScorer
from curriculum import Curriculum
from typing import Any, Callable, Optional
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline, DDPOStableDiffusionPipeline
import torch


@dataclass
class CurriculumTrainerArguments:
    prompt_filename: str = field()
    pretrained_model: str = field(
        default="runwayml/stable-diffusion-v1-5", metadata={"help": "the pretrained model to use"}
    )
    vqa_model: str = field(default="llavav1.6-7b", metadata={"help": "the pretrained model to use"})
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use"})
    hf_hub_model_id: str = field(
        default="ddpo-finetuned-stable-diffusion", metadata={"help": "HuggingFace repo to save model weights to"}
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})
    curriculum_strategy: str = field(default="random", metadata={"help": "The curriculum strategy to use for training."})


class DDPOCurriculumTrainer(DDPOTrainer):
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

    def compute_rewards(self, prompt_image_pairs, is_async=False) -> list[torch.Tensor]:
        rewards = super().compute_rewards(prompt_image_pairs, is_async)
        metadata = {
            "prompt_image_pairs": prompt_image_pairs,
            "rewards": rewards,
            "current_step": self.num_train_timesteps
        }
        target_difficulty = self.curriculum.infer_target_difficulty(metadata=metadata)
        self.update_target_difficulty(target_difficulty)
        return rewards


class DiffusionCurriculumTrainer:
    def __init__(self, curriculum_args: CurriculumTrainerArguments, ddpo_args: DDPOConfig) -> None:
        prompt_loader = CurriculumPromptLoader(prompt_path=curriculum_args.prompt_filename)
        sd_pipeline = DefaultDDPOStableDiffusionPipeline(
            curriculum_args.pretrained_model,
            pretrained_model_revision=curriculum_args.pretrained_revision,
            use_lora=curriculum_args.use_lora,
        )
        scorer_ = VQAScorer(curriculum_args.vqa_model, prompt_loader.set_difficulty)
        self.curriculum = Curriculum(strategy=curriculum_args.curriculum_strategy)
        self._trainer = DDPOCurriculumTrainer(
            curriculum=self.curriculum,
            update_target_difficulty=prompt_loader.set_difficulty,
            config=ddpo_args,
            reward_function=scorer_.calc_score,
            prompt_function=prompt_loader.next,
            sd_pipeline=sd_pipeline,
        )

    def train(self):
        self._trainer.train()