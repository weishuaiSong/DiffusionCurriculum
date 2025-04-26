from typing import Any
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from transformers.pipelines import pipeline
from typing import Callable


def is_answer_match(ans: str, should: str) -> bool:
    return ans.lower() in should.lower()


class VQAScorer:
    def __init__(self, vqa_model_name: str, set_curr_score: Callable[[int], None]) -> None:
        self.vqa_pipeline = pipeline(
            "image-text-to-text", model=vqa_model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.set_curr_score = set_curr_score

    def calc_score(self, images: torch.Tensor, prompts: tuple[str], metadata: tuple[Any]) -> torch.Tensor:
        scores = []
        to_pil = ToPILImage()

        for i, image in enumerate(images):
            qa: list[dict[str, str]] = (
                metadata[i]["qa"]["object"] + metadata[i]["qa"]["relation"] + metadata[i]["qa"]["attribute"]
            )
            if isinstance(image, torch.Tensor):
                image = to_pil(image)
            score = 0
            all_qa = []
            for each_qa in qa:
                all_qa.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image,
                                },
                                {
                                    "type": "text",
                                    "text": each_qa["question"],
                                },
                            ],
                        }
                    ]
                )
            response = self.vqa_pipeline(text=all_qa)  # type: ignore

            for i, resp in enumerate(response):
                answer = resp[0]["generated_text"][-1]["content"]
                score += 1 / len(qa) if is_answer_match(answer, qa[i]["answer"]) else 0
                scores.append(score)
        return np.array(scores), None  # type: ignore
