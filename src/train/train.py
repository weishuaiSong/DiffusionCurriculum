from dataclasses import dataclass, field
import os
from train.ordered_dataloader import CurriculumPromptLoader
from train.scorer import VQAScorer
from train.curriculum import Curriculum
from typing import Any, Callable, Optional
from train.trainer import dpok
from trl.models.modeling_sd_base import DefaultDDPOStableDiffusionPipeline, DDPOStableDiffusionPipeline
from train.trainer.ddpo import DDPOConfig, DDPOTrainer
import torch


@dataclass
class CurriculumTrainerArguments:
    prompt_filename: str = field()
    pretrained_model: str = field(
        default="runwayml/stable-diffusion-v1-5", metadata={"help": "the pretrained model to use"}
    )
    vqa_model: str = field(
        default="llava-hf/llava-v1.6-mistral-7b-hf", metadata={"help": "the pretrained model to use"}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use"})
    hf_hub_model_id: str = field(
        default="ddpo-finetuned-stable-diffusion", metadata={"help": "HuggingFace repo to save model weights to"}
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})
    curriculum_strategy: str = field(
        default="random", metadata={"help": "The curriculum strategy to use for training."}
    )
    rl_algorithm: str = field(
        default="ddpo",
        metadata={"help": "The RL algorithm to use for training, supported algorithms: ddpo, d3po, dpok"},
    )


class DiffusionCurriculumTrainer:
    def __init__(self, curriculum_args: CurriculumTrainerArguments, rl_args: dpok.Config) -> None:
        prompt_loader = CurriculumPromptLoader(prompt_path=curriculum_args.prompt_filename)
        scorer_ = VQAScorer(curriculum_args.vqa_model, prompt_loader.set_difficulty)
        self.curriculum = Curriculum(strategy=curriculum_args.curriculum_strategy)
        self._trainer = dpok.Trainer(
            curriculum=self.curriculum,
            update_target_difficulty=prompt_loader.set_difficulty,
            config=rl_args,
            reward_function=scorer_.calc_score,
            prompt_function=prompt_loader.next,
        )

    def train(self):
        self._trainer.train()
