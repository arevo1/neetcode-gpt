import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        n = len(y_true)
        entropy_sum = 0
        for i, y in enumerate(y_true):
            corr_term = y_pred[i] + 1e-7
            entropy_sum += y*np.log(corr_term) + (1-y)*np.log(1 - corr_term)
        
        return round((-1/n) * entropy_sum, 4) 

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        n = len(y_true)
        c = len(y_true[0])
        entropy_sum = 0
        for i, y in enumerate(y_true):
            j = 0
            while j < c:
                corr_term = y_pred[i][j] + 1e-7
                entropy_sum += y[j] * np.log(corr_term)
                j += 1
        return round((-1/n) * entropy_sum,4) 
