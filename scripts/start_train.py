from train.trainer import d3po
from train.train import CurriculumTrainerArguments, DiffusionCurriculumTrainer
from transformers.hf_argparser import HfArgumentParser


def main():
    parser = HfArgumentParser((CurriculumTrainerArguments, d3po.Config))  # type: ignore
    curriculum_args, rl_args = parser.parse_args_into_dataclasses()

    trainer = DiffusionCurriculumTrainer(curriculum_args, rl_args)
    trainer.train()


if __name__ == "__main__":
    main()
