import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = np.dot(x, w) + b
        y_hat =  1 / (1 + np.exp(-z))
        return (np.round((y_hat - y_true)*y_hat*(1-y_hat)*x, 5)
        , np.round((y_hat - y_true)*y_hat*(1-y_hat), 5))
        