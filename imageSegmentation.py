import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import argparse

# python imageSegmentation.py 4 output/hawk.jpg output/hawk-out4.jpg

def initialize_centers_plusplus(data, K):
    centers = [data[np.random.randint(data.shape[0]), :]]
    for _ in range(1, K):
        distances = np.array([min([np.inner(c - x, c - x) for c in centers]) for x in data])
        probabilities = distances / distances.sum()
        cumulative_probabilities = probabilities.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                centers.append(data[j])
                break
    return np.array(centers)


def k_means(data, K, iterations):
    centers = initialize_centers_plusplus(data, K)
    for iteration in range(iterations):
        # Assignment step
        distances = np.sqrt(((data - centers[:, np.newaxis]) ** 2).sum(axis=2))
        closest_cluster = np.argmin(distances, axis=0)

        # Update step
        for i in range(K):
            points_in_cluster = data[closest_cluster == i]
            if points_in_cluster.size:
                centers[i] = np.mean(points_in_cluster, axis=0)

    return closest_cluster, centers


def segment_image(image_path, K, iterations, output_path):
    image = Image.open(image_path)
    imageW, imageH = image.size
    data = np.array(image).reshape((-1, 3))

    # adding position to the features
    x, y = np.meshgrid(range(imageW), range(imageH))
    data_with_pos = np.hstack((data, x.flatten().reshape(-1, 1), y.flatten().reshape(-1, 1)))

    # scaling features
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_with_pos)

    pixel_labels, centers = k_means(data_scaled, K, iterations)

    # assigning new colors
    centers_colors = centers[:, :3]
    new_colors = centers_colors[pixel_labels].reshape(imageH, imageW, 3)

    new_image_data = (new_colors * 255).astype(np.uint8)
    new_image = Image.fromarray(new_image_data)
    new_image.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Segmentation using K-means')
    parser.add_argument('K', type=int, help='Number of clusters')
    parser.add_argument('input_image', type=str, help='Input image path')
    parser.add_argument('output_image', type=str, help='Output image path')
    args = parser.parse_args()

    K = args.K
    inputName = args.input_image
    outputName = args.output_image
    iterations = 5

    segment_image(inputName, K, iterations, outputName)
