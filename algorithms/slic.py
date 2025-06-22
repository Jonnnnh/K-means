import numpy as np
from skimage.segmentation import slic
from .base import Segmenter
from PIL import Image


class SLICSeg(Segmenter):
    def __init__(self, n_segments=100, compactness=10.0, max_dim=None):
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_dim = max_dim

    def segment(self, img_arr, **kwargs):
        H, W = img_arr.shape[:2]
        if self.max_dim and max(H, W) > self.max_dim:
            scale = self.max_dim / max(H, W)
            img_arr = np.array(
                Image.fromarray(img_arr).resize(
                    (int(W * scale), int(H * scale)),
                    Image.LANCZOS
                )
            )
            H, W = img_arr.shape[:2]

        channel_axis = -1 if img_arr.ndim == 3 else None
        segments = slic(
            img_arr,
            n_segments=self.n_segments,
            compactness=self.compactness,
            start_label=0,
            channel_axis=channel_axis
        )

        return segments.flatten(), (H, W)
