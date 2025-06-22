import gradio as gr
import numpy as np
from PIL import Image
from skimage.color import label2rgb

from utils import preprocess_image, postprocess_and_save
from algorithms.kmeans_pp import KMeansPP
from algorithms.minibatch_kmeans import MiniBatchKMeansSeg
from algorithms.superpixel_kmeans import SuperpixelKMeans
from algorithms.gmm import GMM
from algorithms.meanshift import FastMeanShiftSeg
from algorithms.slic import SLICSeg


def segment(img, method, K, iters, batch, n_sp,
            quantile, n_samples, n_seg, compactness, max_dim, cov_type):
    img_arr = np.array(img)
    H, W = img_arr.shape[:2]

    if method == 'slic':
        segmenter = SLICSeg(n_segments=n_seg,
                            compactness=compactness,
                            max_dim=max_dim)
        labels, shape = segmenter.segment(img_arr)
        h_new, w_new = shape
        if (h_new, w_new) != img_arr.shape[:2]:
            bg = np.array(Image.fromarray(img_arr).resize((w_new, h_new),
                                                          Image.LANCZOS))
        else:
            bg = img_arr
        seg = label2rgb(labels.reshape(h_new, w_new), bg, kind='avg')
        return Image.fromarray((seg * 255).astype(np.uint8))

    if method == 'spkmeans':
        segmenter = SuperpixelKMeans(
            n_sp=n_sp,
            K=K,
            compactness=compactness,
            max_dim=max_dim
        )
        labels, shape = segmenter.segment(img_arr)
        h_new, w_new = shape
        if (h_new, w_new) != img_arr.shape[:2]:
            bg = np.array(Image.fromarray(img_arr)
                          .resize((w_new, h_new), Image.LANCZOS))
        else:
            bg = img_arr

        seg = label2rgb(labels.reshape(h_new, w_new), bg, kind='avg')
        return Image.fromarray((seg * 255).astype(np.uint8))

    data, meta = preprocess_image(img, max_dim)
    if method == 'kmeans':
        segmenter = KMeansPP(K=K, iterations=iters)
    elif method == 'minibatch':
        segmenter = MiniBatchKMeansSeg(K=K,
                                       batch_size=batch,
                                       max_iter=iters)
    elif method == 'gmm':
        segmenter = GMM(K=K,
                        covariance_type=cov_type)
    else:
        segmenter = FastMeanShiftSeg(quantile=quantile,
                                     n_samples=n_samples,
                                     bin_seeding=True)

    labels, centers = segmenter.segment(data)
    return postprocess_and_save(labels, centers, meta, None)


iface = gr.Interface(
    fn=segment,
    inputs=[
        gr.Image(type="pil", label="Изображение"),
        gr.Radio(['kmeans', 'minibatch', 'spkmeans', 'gmm', 'meanshift', 'slic'],
                 value='kmeans', label='Метод'),
        gr.Slider(2, 10, 4, 1, label='K'),
        gr.Slider(1, 200, 10, 1, label='Max iterations'),
        gr.Slider(100, 5000, 1000, 100, label='Batch size (MiniBatch)'),
        gr.Slider(100, 5000, 1000, 100, label='Superpixels (SPKMeans)'),
        gr.Slider(0.01, 1.0, 0.2, 0.01, label='Quantile (MeanShift)'),
        gr.Slider(1000, 10000, 5000, 500, label='Samples for bw est.'),
        gr.Slider(50, 1000, 100, 50, label='SLIC n_segments'),
        gr.Slider(1, 20, 10, 1, label='SLIC compactness'),
        gr.Slider(100, 2000, 800, 100, label='Max dimension'),
        gr.Dropdown(['diag', 'tied', 'full', 'spherical'],
                    value='diag',
                    label='GMM covariance type')
    ],
    outputs=gr.Image(type="pil", label="Результат"),
    title="Fast Image Segmentation",
)

iface.launch()
