from typing import Any
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from transformers.pipelines import pipeline
from bert_score import score as calc_bert_score


class VQAScorer:
    def __init__(self, vqa_model_name: str) -> None:
        self.vqa_pipeline = pipeline("image-text-to-text", model=vqa_model_name)

    def calc_score(self, images: torch.Tensor, prompts: tuple[str], metadata: tuple[Any]) -> torch.Tensor:
        scores = []
        to_pil = ToPILImage()

        for i, image in enumerate(images):
            qa: list[dict[str, str]] = metadata[i]["qa"]
            if isinstance(image, torch.Tensor):
                image = to_pil(image)
            response = self.vqa_pipeline(
                text=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {
                                "type": "text",
                                "text": qa[0]["question"],
                            },
                        ],
                    }
                ]
            )  # type: ignore
            answer = response[0]["generated_text"]
            _, _, bert_scores = calc_bert_score([answer], [qa[0]["answer"]], lang="en", verbose=True)
            reward_score = bert_scores.mean().item() * 10  # type: ignore
            score = np.array(reward_score)

            scores.append(score)
        return np.array(scores)  # type: ignore
