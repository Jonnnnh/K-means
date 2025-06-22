import numpy as np
from skimage.segmentation import slic
from PIL import Image
from .base import Segmenter
from .minibatch_kmeans import MiniBatchKMeansSeg

class SuperpixelKMeans(Segmenter):
    def __init__(self, n_sp=1000, K=4, compactness=10, max_dim=None):
        self.n_sp = n_sp
        self.K = K
        self.compactness = compactness
        self.max_dim = max_dim

    def segment(self, img_arr, **kwargs):
        H, W = img_arr.shape[:2]
        if self.max_dim and max(H, W) > self.max_dim:
            scale = self.max_dim / max(H, W)
            img_arr = np.array(
                Image.fromarray(img_arr).resize(
                    (int(W * scale), int(H * scale)), Image.LANCZOS)
            )
            H, W = img_arr.shape[:2]
        channel_axis = -1 if img_arr.ndim == 3 else None
        segments = slic(img_arr,
                        n_segments=self.n_sp,
                        compactness=self.compactness,
                        channel_axis=channel_axis,
                        start_label=0)
        flat = segments.ravel()
        pixels = (img_arr.reshape(-1, 3) if img_arr.ndim == 3
                  else np.stack([img_arr.ravel()]*3, axis=1))
        counts = np.bincount(flat)
        sums = np.vstack([np.bincount(flat, weights=pixels[:, ch])
                          for ch in range(3)])
        sp_feats = (sums / counts).T
        labels_sp, _ = MiniBatchKMeansSeg(self.K).segment(sp_feats)
        label_map = labels_sp
        labels_pix = label_map[flat]

        return labels_pix, (H, W)