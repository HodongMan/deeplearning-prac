
"""
Discounted Cumulative Gain @ R is
https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
"""

import numpy as np
from scipy.sparse import csr_matrix


def ndcg(x_true: csr_matrix, X_top_k: np.array, R=100) -> np.array:

    penalties       = 1. / np.log2(np.arange(2, R + 2))
    selected        = np.take_along_axis(X_true, X_top_k[:, :R], axis=-1)

    DCG             = selected * penalties

    cpenalties      = np.empty(R + 1)
    np.cumsum(penalties, out=cpenalties[1:])

    cpenalties[0]   = 0
    maxhit          = np.minimum(X_true.getnnz(axis=1), R)
    IDCG            = cpenalties[maxhit]

    return DCG / IDCG
