from typing import Any
import torch

# TODO: change the *** name
imageqa_models = {
    "llavav1.5-7b": (LLaVA, "llava-hf/llava-1.5-7b-hf"),
    "llavav1.5-13b": ("LLaVA", "llava-hf/llava-1.5-13b-hf"),
    "llavav1.6-7b": ("LLaVA", "llava-hf/llava-v1.6-vicuna-7b-hf"),
    "llavav1.6-13b": ("LLaVA", "llava-hf/llava-v1.6-vicuna-13b-hf"),
    "qwenvl": ("QwenVL", "Qwen/Qwen-VL"),
    "qwenvl-chat": ("QwenVLChat", "Qwen/Qwen-VL-Chat"),
    "internvl-chat-v1.5": ("InternVLChat", "failspy/InternVL-Chat-V1-5-quantable"),
    "idefics2-8b": ("IDEFICS2", "HuggingFaceM4/idefics2-8b"),
    "llavav1.5-7b-100-templated": ("LLaVA", "shijianS01/llava-v1.5-7b-lora-100-templated"),
    "llavav1.5-7b-1k-templated": ("LLaVA", "shijianS01/llava-v1.5-7b-lora-1k-templated"),
    "llavav1.5-7b-5k-templated": ("LLaVA", "shijianS01/llava-v1.5-7b-lora-5k-templated"),
    "llavav1.5-7b-10k-templated": ("LLaVA", "shijianS01/llava-v1.5-7b-lora-10k-templated"),
    "llavav1.5-7b-15k-templated": ("LLaVA", "shijianS01/llava-v1.5-7b-lora-15k-templated"),
    "llavav1.5-13b-100-templated": ("LLaVA", "shijianS01/llava-v1.5-13b-lora-100-templated"),
    "llavav1.5-13b-1k-templated": ("LLaVA", "shijianS01/llava-v1.5-13b-lora-1k-templated"),
    "llavav1.5-13b-5k-templated": ("LLaVA", "shijianS01/llava-v1.5-13b-lora-5k-templated"),
    "llavav1.5-13b-10k-templated": ("LLaVA", "shijianS01/llava-v1.5-13b-lora-10k-templated"),
    "llavav1.5-13b-15k-templated": ("LLaVA", "shijianS01/llava-v1.5-13b-lora-15k-templated"),
}


class VQAScorer:
    def __init__(self, vqa_model_name: str) -> None:
        self.vqa_model = ImageQAModel(
            model_name="llavav1.6-7b",
            torch_device="auto",
            enable_choice_search=True,
            precision=torch.float16,
        )

    def calc_score(self, images: torch.Tensor, prompts: tuple[str], metadata: tuple[Any]) -> torch.Tensor: ...
