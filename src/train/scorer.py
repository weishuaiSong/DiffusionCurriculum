from typing import Any
import torch
from transformers.models.llava_next import LlavaNextProcessor, LlavaNextForConditionalGeneration


class ImageQAModel:
    def __init__(self, model_name: str, torch_device: str, enable_choice_search: bool, precision: torch.dtype) -> None:
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=precision, low_cpu_mem_usage=True, device_map=torch_device
        ).eval()
        self.processor = LlavaNextProcessor.from_pretrained(model_name, device_map=torch_device)


class VQAScorer:
    def __init__(self, vqa_model_name: str) -> None:
        self.vqa_model = ImageQAModel(
            model_name="llavav1.6-7b",
            torch_device="auto",
            enable_choice_search=True,
            precision=torch.float16,
        )

    def calc_score(self, images: torch.Tensor, prompts: tuple[str], metadata: tuple[Any]) -> torch.Tensor: ...
