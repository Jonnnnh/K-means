from sklearn.cluster import MiniBatchKMeans
from .base import Segmenter


class MiniBatchKMeansSeg(Segmenter):
    def __init__(self, K, batch_size=1000, max_iter=100):
        self.K = K
        self.batch_size = batch_size
        self.max_iter = max_iter

    def segment(self, data, **kwargs):
        mbk = MiniBatchKMeans(n_clusters=self.K,
                              batch_size=self.batch_size,
                              max_iter=self.max_iter,
                              random_state=0)
        labels = mbk.fit_predict(data)
        return labels, mbk.cluster_centers_
