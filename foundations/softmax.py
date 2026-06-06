import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        max_val = max(z)
        denom = np.sum([np.exp(x - max_val) for x in z])
        return np.array([round(np.exp(x - max_val)/denom, 4) for x in z])
