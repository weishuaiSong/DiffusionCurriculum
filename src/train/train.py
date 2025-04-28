from dataclasses import dataclass, field
from train.ordered_dataloader import CurriculumPromptLoader
from train.scorer import VQAScorer
from train.curriculum import Curriculum
from train.trainer import dpok, d3po, ddpo


@dataclass
class CurriculumTrainerArguments:
    prompt_filename: str = field()
    vqa_model: str = field(
        default="llava-hf/llava-v1.6-mistral-7b-hf", metadata={"help": "the pretrained model to use"}
    )
    curriculum_strategy: str = field(
        default="random", metadata={"help": "The curriculum strategy to use for training."}
    )
    rl_algorithm: str = field(
        default="dpok",
        metadata={"help": "The RL algorithm to use for training, supported algorithms: ddpo, d3po, dpok"},
    )


class DiffusionCurriculumTrainer:
    def __init__(self, curriculum_args: CurriculumTrainerArguments, rl_args) -> None:
        prompt_loader = CurriculumPromptLoader(prompt_path=curriculum_args.prompt_filename)
        scorer_ = VQAScorer(curriculum_args.vqa_model, prompt_loader.set_difficulty)
        self.curriculum = Curriculum(strategy=curriculum_args.curriculum_strategy)

        # 根据选定的RL算法初始化相应的训练器
        if curriculum_args.rl_algorithm == "dpok":
            self._trainer = dpok.Trainer(
                curriculum=self.curriculum,
                update_target_difficulty=prompt_loader.set_difficulty,
                config=rl_args,
                reward_function=scorer_.calc_score,
                prompt_function=prompt_loader.next,
            )
        elif curriculum_args.rl_algorithm == "d3po":
            self._trainer = d3po.Trainer(
                curriculum=self.curriculum,
                update_target_difficulty=prompt_loader.set_difficulty,
                config=rl_args,
                reward_function=scorer_.calc_score,
                prompt_function=prompt_loader.next,
            )
        elif curriculum_args.rl_algorithm == "ddpo":
            self._trainer = ddpo.Trainer(
                curriculum=self.curriculum,
                update_target_difficulty=prompt_loader.set_difficulty,
                config=rl_args,
                reward_function=scorer_.calc_score,
                prompt_function=prompt_loader.next,
            )
        else:
            raise ValueError(f"不支持的RL算法: {curriculum_args.rl_algorithm}，支持的算法有: ddpo, d3po, dpok")

    def train(self):
        self._trainer.train()
