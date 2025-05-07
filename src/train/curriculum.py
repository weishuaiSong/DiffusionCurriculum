from typing import Any
import random


class Curriculum:
    def __init__(self, strategy: str = "random"):
        self.strategy = strategy

    def infer_target_difficulty(self, metadata: dict[str, Any]) -> int:
        if self.strategy == "random":
            # TODO: 需要传进来难度scale，或者我们后面定好
            return random.randint(3, 22)
        elif self.strategy == "reward":
            return self._reward_based_infer(metadata)
        elif self.strategy == "timestep":
            return self._timestep_based_infer(metadata)
        else:
            raise NotImplementedError("Not implemented yet")

    def _reward_based_infer(self, metadata: dict[str, Any]) -> int:
        pass

    def _timestep_based_infer(self, metadata: dict[str, Any]) -> int:
        return min(max(0, metadata["current_step"] // 7) + 3, 22)
