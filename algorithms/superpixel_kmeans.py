import numpy as np
from skimage.segmentation import slic
from .base import Segmenter
from .minibatch_kmeans import MiniBatchKMeansSeg


class SuperpixelKMeans(Segmenter):
    def __init__(self, n_sp=1000, K=4, compactness=10):
        self.n_sp = n_sp
        self.K = K
        self.compactness = compactness

    def segment(self, img_arr, **kwargs):
        channel_axis = -1 if img_arr.ndim == 3 else None
        segments = slic(
            img_arr,
            n_segments=self.n_sp,
            compactness=self.compactness,
            channel_axis=channel_axis
        )
        H, W = segments.shape
        flat = segments.flatten()
        if img_arr.ndim == 3:
            pixels = img_arr.reshape(-1, 3)
        else:
            pixels = np.stack([img_arr.flatten()] * 3, axis=1)

        unique_ids = np.unique(flat)
        sp_feats = np.vstack([pixels[flat == sid].mean(axis=0) for sid in unique_ids])

        labels_sp, centers = MiniBatchKMeansSeg(self.K).segment(sp_feats)

        max_id = unique_ids.max()
        label_map = np.zeros(max_id + 1, dtype=int)
        label_map[unique_ids] = labels_sp

        labels_pix = label_map[flat]
        return labels_pix, centers
