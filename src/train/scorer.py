from collections import defaultdict
from typing import Any
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from transformers import Pipeline
from transformers.pipelines import pipeline
from typing import Callable


def is_answer_match(ans: str, should: str) -> bool:
    return should.lower().strip() in ans.lower()


class VQAScorer:
    def __init__(self, set_curr_score: Callable[[int], None]) -> None:
        self.set_curr_score = set_curr_score

    def calc_score(
            self, vqa_pipeline: Pipeline, images: torch.Tensor, prompts: tuple[str], metadata: tuple[Any]
    ):
        batch_size = len(images)
        scores = [0.0] * len(images)
        to_pil = ToPILImage()

        vqa_samples = []

        for i, image in enumerate(images):
            all_qa: list[dict[str, str]] = (
                    metadata[i]["qa"]["object"] + metadata[i]["qa"]["relation"] + metadata[i]["qa"]["attribute"]
            )
            if isinstance(image, torch.Tensor):
                pil_image = to_pil(image.to(torch.float))
            for each_qa in all_qa:
                vqa_samples.append(
                    (
                        [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": pil_image,
                                    },
                                    {
                                        "type": "text",
                                        "text": f"Based on the image, answer the question. Only output yes or no.\nQuestion: {each_qa["question"]}\nAnswer:",
                                    },
                                ],
                            }
                        ],
                        each_qa["answer"],
                        len(all_qa),
                        i,
                    )
                )

        # calc reward in batch
        for i in range(0, len(vqa_samples), batch_size):
            q_with_image, answers, qa_lens, img_indices = zip(*vqa_samples[i: i + batch_size])
            responses = vqa_pipeline(text=q_with_image)  # type: ignore

            for response, answer, qa_len, img_idx in zip(responses, answers, qa_lens, img_indices):
                generated_answer = response[0]["generated_text"][-1]["content"]
                scores[img_idx] += (1 / qa_len) if is_answer_match(generated_answer, answer) else 0

        return np.array(scores), None  # type: ignore
