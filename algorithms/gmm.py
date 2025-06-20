from sklearn.mixture import GaussianMixture
from .base import Segmenter


class GMM(Segmenter):
    def __init__(self, K, covariance_type='tied'):
        self.K = K
        self.covariance_type = covariance_type

    def segment(self, data, **kwargs):
        gmm = GaussianMixture(n_components=self.K,
                              covariance_type=self.covariance_type)
        labels = gmm.fit_predict(data)
        return labels, gmm.means_
