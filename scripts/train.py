from train.trainer import CurriculumTrainerArguments, DiffusionCurriculumTrainer
from trl import DDPOConfig
from transformers.hf_argparser import HfArgumentParser


def main():
    parser = HfArgumentParser((CurriculumTrainerArguments, DDPOConfig))  # type: ignore
    script_args, ddpo_args = parser.parse_args_into_dataclasses()
    script_args: CurriculumTrainerArguments
    ddpo_args: DDPOConfig

    trainer = DiffusionCurriculumTrainer(script_args, ddpo_args)
    trainer.train()


if __name__ == "__main__":
    main()
