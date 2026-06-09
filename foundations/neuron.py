import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        z = np.dot(x, w) + b

        if activation == "sigmoid":
            return round((1/(1+np.exp(-z))),5)
        elif activation.lower() == "relu":
            return round(float(max(0, z)),5)
