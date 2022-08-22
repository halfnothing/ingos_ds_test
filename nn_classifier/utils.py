import numpy as np
from numpy import ndarray


def distance(a: np.array, b: np.array) -> ndarray:
    return np.sum((a - b) ** 2)
