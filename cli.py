import argparse
from PIL import Image
import numpy as np
from utils import preprocess_image, postprocess_and_save
from importlib import import_module

METHODS = {
    'kmeans': 'image_segmentation.algorithms.kmeans_pp:KMeansPP',
    'minibatch': 'image_segmentation.algorithms.minibatch_kmeans:MiniBatchKMeansSeg',
    'spkmeans': 'image_segmentation.algorithms.superpixel_kmeans:SuperpixelKMeans',
    'gmm': 'image_segmentation.algorithms.gmm:GMM',
    'meanshift': 'image_segmentation.algorithms.meanshift:MeanShiftSeg',
    'slic': 'image_segmentation.algorithms.slic:SLICSeg'
}


def load_segmenter(spec, **kwargs):
    module, cls = spec.split(':')
    mod = import_module(module)
    return getattr(mod, cls)(**kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS, default='kmeans')
    parser.add_argument('-K', type=int, default=4)
    parser.add_argument('--max-dim', type=int, default=None)
    parser.add_argument('--batch', type=int, default=1000)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--n-sp', type=int, default=1000)
    parser.add_argument('--bandwidth', type=float, default=None)
    parser.add_argument('--n-segments', type=int, default=100)
    parser.add_argument('--compactness', type=float, default=10.0)
    parser.add_argument('input_image')
    parser.add_argument('output_image')
    args = parser.parse_args()

    img = Image.open(args.input_image)
    data = None
    if args.method == 'slic':
        data = np.array(img)
        seg = load_segmenter(METHODS['slic'],
                             n_segments=args.n_segments,
                             compactness=args.compactness)
    else:
        data, meta = preprocess_image(img, args.max_dim)
        if args.method == 'kmeans':
            seg = load_segmenter(METHODS['kmeans'],
                                 K=args.K,
                                 iterations=args.iters)
        elif args.method == 'minibatch':
            seg = load_segmenter(METHODS['minibatch'],
                                 K=args.K,
                                 batch_size=args.batch,
                                 max_iter=args.iters)
        elif args.method == 'spkmeans':
            data = np.array(img)
            seg = load_segmenter(METHODS['spkmeans'],
                                 n_sp=args.n_sp,
                                 K=args.K)
        elif args.method == 'gmm':
            seg = load_segmenter(METHODS['gmm'], K=args.K)
        else:
            seg = load_segmenter(METHODS['meanshift'], bandwidth=args.bandwidth)
    labels, centers = seg.segment(data)
    if args.method == 'slic' or args.method == 'spkmeans':
        from skimage.color import label2rgb

        arr = np.array(img)
        seg_img = label2rgb(labels.reshape(img.size[1], img.size[0]), arr)
        out = (seg_img * 255).astype(np.uint8)
        Image.fromarray(out).save(args.output_image)
    else:
        postprocess_and_save(labels, centers, meta, args.output_image)
