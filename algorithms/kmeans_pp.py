import numpy as np
from .base import Segmenter


class KMeansPP(Segmenter):
    def __init__(self, K, iterations=10, tol=1e-3):
        self.K = K
        self.iterations = iterations
        self.tol = tol

    def _init_plusplus(self, data):
        centers = [data[np.random.randint(len(data))]]
        for _ in range(1, self.K):
            dist2 = np.min([((data - c) ** 2).sum(1) for c in centers], axis=0)
            probs = dist2 / dist2.sum()
            i = np.searchsorted(probs.cumsum(), np.random.rand())
            centers.append(data[i])
        return np.vstack(centers)

    def segment(self, data, **kwargs):
        centers = self._init_plusplus(data)
        for _ in range(self.iterations):
            dists = np.linalg.norm(data[:, None] - centers[None, :], axis=2)
            labels = dists.argmin(1)
            new_centers = np.vstack(
                [data[labels == i].mean(0) if np.any(labels == i) else centers[i] for i in range(self.K)])
            if np.linalg.norm(new_centers - centers) < self.tol:
                break
            centers = new_centers
        return labels, centers
