from typing import Any
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CurriculumPromptLoader:
    def __init__(self, prompt_path: str) -> None:
        self.difficulty_to_prompts: dict[int, list[dict[str, Any]]] = {}
        self.difficulty_to_prompts_idx: dict[int, int] = {}
        # TODO: 没用calculator，直接从prompt_path中读取，未来可能修改
        self.prompt_path = Path(prompt_path)
        self.current_difficulty = 3
        for difficulty_str, prompts in json.loads(self.prompt_path.read_text()).items():
            self.difficulty_to_prompts[self._extract_difficulty(difficulty_str)] = prompts
            self.difficulty_to_prompts_idx[self._extract_difficulty(difficulty_str)] = 0

    def _extract_difficulty(self, difficulty_str: str) -> int:
        return int(difficulty_str.split("_")[-1])

    def next(self) -> tuple[str, Any]:
        if self.difficulty_to_prompts_idx[self.current_difficulty] >= len(
            self.difficulty_to_prompts[self.current_difficulty]
        ):
            logger.warning(f"difficulty {self.current_difficulty} has no more prompts, reset to 0")
            self.difficulty_to_prompts_idx[self.current_difficulty] = 0
        prompt = self.difficulty_to_prompts[self.current_difficulty][
            self.difficulty_to_prompts_idx[self.current_difficulty]
        ]
        self.difficulty_to_prompts_idx[self.current_difficulty] += 1
        return prompt["prompt"], prompt

    def set_difficulty(self, difficulty: int) -> None:
        self.current_difficulty = difficulty
