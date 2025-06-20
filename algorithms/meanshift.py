import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from .base import Segmenter


class FastMeanShiftSeg(Segmenter):
    def __init__(self, quantile=0.2, n_samples=5000, bin_seeding=True):
        self.quantile = quantile
        self.n_samples = n_samples
        self.bin_seeding = bin_seeding

    def segment(self, data, **kwargs):
        N = data.shape[0]
        if N > self.n_samples:
            idx = np.random.choice(N, self.n_samples, replace=False)
            sample = data[idx]
        else:
            sample = data

        bw = estimate_bandwidth(sample,
                                quantile=self.quantile,
                                n_samples=len(sample))

        ms = MeanShift(bandwidth=bw,
                       bin_seeding=self.bin_seeding)
        labels = ms.fit_predict(data)
        return labels, ms.cluster_centers_
