from abc import ABC, abstractmethod
import numpy as np


class SkelAlgorithm(ABC):
    @abstractmethod
    def compute(self, img: np.ndarray) -> np.ndarray:
        pass