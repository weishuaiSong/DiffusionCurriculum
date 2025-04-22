from dataclasses import dataclass, field
from train.ordered_dataloader import CurriculumPromptLoader
from train.scorer import VQAScorer
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline


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
    hf_hub_aesthetic_model_id: str = field(
        default="trl-lib/ddpo-aesthetic-predictor",
        metadata={"help": "HuggingFace model ID for aesthetic scorer model weights"},
    )
    hf_hub_aesthetic_model_filename: str = field(
        default="aesthetic-model.pth",
        metadata={"help": "HuggingFace model filename for aesthetic scorer model weights"},
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})


class DiffusionCurriculumTrainer:
    def __init__(self, curriculum_args: CurriculumTrainerArguments, ddpo_args: DDPOConfig) -> None:
        prompt_loader = CurriculumPromptLoader(prompt_path=curriculum_args.prompt_filename)
        sd_pipeline = DefaultDDPOStableDiffusionPipeline(
            curriculum_args.pretrained_model,
            pretrained_model_revision=curriculum_args.pretrained_revision,
            use_lora=curriculum_args.use_lora,
        )
        scorer_ = VQAScorer(curriculum_args.vqa_model)
        self._trainer = DDPOTrainer(
            ddpo_args,
            scorer_.calc_score,
            prompt_loader.next,
            sd_pipeline,
            image_samples_hook=image_outputs_logger,
        )

    def train(self):
        self._trainer.train()
