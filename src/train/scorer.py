from typing import Any
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from transformers.pipelines import pipeline
from typing import Callable


def is_answer_match(ans: str, should: str) -> bool:
    return should.lower().strip() in ans.lower()


class VQAScorer:
    def __init__(self, vqa_model_name: str, set_curr_score: Callable[[int], None], batch_size: int) -> None:
        self.vqa_pipeline = pipeline(
            "image-text-to-text",
            model=vqa_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            batch_size=batch_size,
        )
        self.set_curr_score = set_curr_score

    def calc_score(self, images: torch.Tensor, prompts: tuple[str], metadata: tuple[Any]) -> torch.Tensor:
        batch_size = len(images)
        scores = []
        to_pil = ToPILImage()

        all_qa = []
        for i, image in enumerate(images):
            qa: list[dict[str, str]] = (
                metadata[i]["qa"]["object"] + metadata[i]["qa"]["relation"] + metadata[i]["qa"]["attribute"]
            )
            if isinstance(image, torch.Tensor):
                image = to_pil(image.to(torch.float))
            for each_qa in qa:
                all_qa.append(
                    (
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
                        ],
                        each_qa["answer"],
                        len(qa),
                    )
                )
        for i in range(0, len(all_qa), batch_size):
            q_with_contents, answers, qa_lens = zip(*all_qa[i : i + batch_size])
            response = self.vqa_pipeline(text=q_with_contents)  # type: ignore

            score = 0
            for i, resp in enumerate(response):
                answer = resp[0]["generated_text"][-1]["content"]
                score += 1 / qa_lens[i] if is_answer_match(answer, answers[i]) else 0
            scores.append(score)
        return np.array(scores), None  # type: ignore
