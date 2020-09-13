"""
https://arxiv.org/pdf/1802.05814.pdf, chapter 4.2
"""

import numpy as np
from scipy.sparse import csr_matrix

def recall(X_true: csr_matrix, X_top_k: np.array, R=100) -> np.array:

    selected    = np.take_along_axis(X_true, X_top_k[:, :R], axis=-1)
    hit         = selected.sum(axis=-1)

    maxhit      = np.minimum(X_true.getnnz(axis=1), R)

    return np.squeeze(np.asarray(hit)) / maxhit