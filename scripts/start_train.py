from train.trainer import dpok, d3po, ddpo
from train.train import CurriculumTrainerArguments, DiffusionCurriculumTrainer
from transformers.hf_argparser import HfArgumentParser
import sys


def main():
    # 首先只解析课程学习参数，以确定使用哪个RL算法
    parser = HfArgumentParser(CurriculumTrainerArguments)
    curriculum_args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # 重置sys.argv以包含未知参数，以便后续解析
    sys.argv = [sys.argv[0]] + unknown_args

    # 根据选择的RL算法选择相应的Config类
    if curriculum_args.rl_algorithm == "dpok":
        ConfigClass = dpok.Config
    elif curriculum_args.rl_algorithm == "d3po":
        ConfigClass = d3po.Config
    elif curriculum_args.rl_algorithm == "ddpo":
        ConfigClass = ddpo.Config
    else:
        raise ValueError(f"不支持的RL算法: {curriculum_args.rl_algorithm}，支持的算法有: ddpo, d3po, dpok")

    # 解析RL特定参数
    parser = HfArgumentParser(ConfigClass)
    rl_args = parser.parse_args_into_dataclasses()[0]

    trainer = DiffusionCurriculumTrainer(curriculum_args, rl_args)
    trainer.train()


if __name__ == "__main__":
    main()
