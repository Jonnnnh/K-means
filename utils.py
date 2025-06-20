import numpy as np
from sklearn.preprocessing import MinMaxScaler
from PIL import Image


def resize_image(image: Image.Image, max_dim: int):
    W, H = image.size
    scale = max_dim / max(W, H)
    if scale < 1:
        return image.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    return image


def preprocess_image(image: Image.Image, max_dim=None):
    if max_dim:
        W, H = image.size
        scale = max_dim / max(W, H)
        if scale < 1:
            image = image.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    arr = np.array(image)
    H, W = arr.shape[:2]

    rgb = arr.reshape(-1, 3)
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    coords = np.stack([x.flatten() / W, y.flatten() / H], axis=1)

    rgb_scaler = MinMaxScaler()
    rgb_scaled = rgb_scaler.fit_transform(rgb)

    data = np.hstack([rgb_scaled, coords])
    meta = {'shape': (H, W), 'rgb_scaler': rgb_scaler}
    return data, meta


def postprocess_and_save(labels, centers, meta, output_path=None):
    H, W = meta['shape']
    rgb_scaler = meta['rgb_scaler']

    orig_rgb = rgb_scaler.inverse_transform(centers[:, :3])
    colors = np.clip(orig_rgb, 0, 255).astype(np.uint8)

    img_arr = colors[labels].reshape(H, W, 3)
    img = Image.fromarray(img_arr)
    if output_path:
        img.save(output_path)
    return img
