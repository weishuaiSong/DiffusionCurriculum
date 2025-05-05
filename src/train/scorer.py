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
    ) -> torch.Tensor:
        batch_size = len(images)
        scores = []
        to_pil = ToPILImage()

        all_qa = []
        # generate qa for vqa
        for i, image in enumerate(images):
            raw_image = image
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
                        raw_image,
                    )
                )

        image_to_scores = defaultdict(int)
        # calc reward in batch
        for i in range(0, len(all_qa), batch_size):
            q_with_contents, answers, qa_lens, raw_image = zip(*all_qa[i : i + batch_size])
            response = vqa_pipeline(text=q_with_contents)  # type: ignore

            for i, resp in enumerate(response):
                answer = resp[0]["generated_text"][-1]["content"]
                image_to_scores[raw_image] += (1 / qa_lens[i]) if is_answer_match(answer, answers[i]) else 0

        # return in same order as input
        for image in images:
            scores.append(image_to_scores[image])
        return np.array(scores), None  # type: ignore
