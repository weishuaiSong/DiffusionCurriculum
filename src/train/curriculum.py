from typing import Any
import numpy as np
import random


class Curriculum:
    def __init__(self, eta: float, alpha: float, beta: float, strategy: str = "random"):
        self.strategy = strategy
        self.eta = eta
        self.alpha = alpha
        self.beta = beta

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
        result_difficulty = metadata["difficulty"] + self.eta * np.tanh(self.alpha * metadata["reward"] - self.beta)
        return min(max(0, result_difficulty) + 3, 22)

    def _timestep_based_infer(self, metadata: dict[str, Any]) -> int:
        return min(max(0, metadata["current_step"] // 7) + 3, 22)
