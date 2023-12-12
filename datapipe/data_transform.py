import numpy as np
import os
from scipy.ndimage import map_coordinates
import warnings
from collections import namedtuple

GaussianParameters = namedtuple('GaussianParameters', ['alpha_dirs', 'mu_dirs', 'sigma_dirs', 'rotation_dirs'])

def gaussian(x, mu, sigma, alpha, epsilon=1e-8):
    """
    Computes a Gaussian function.
    """

    dist = np.power(x - mu[:, None, None], 2)
    dist = np.divide(dist, 2 * sigma[:, None, None] ** 2 + epsilon)
    dist = np.sum(dist, axis=0)
    y = alpha * np.exp(-dist)

    return y




def gaussian_shift(coordinates, image_center, alpha, mu, sigma, dimension, rotation=None, lensing=False):
    """
    Applies a Gaussian transform to an image.
    """
    mu_normalized = mu + image_center
    shift = gaussian(coordinates, mu_normalized, sigma, alpha)

    if lensing:
        sign = np.where(coordinates[dimension] < mu_normalized[dimension], -1., 1.)
        shift = shift * sign

    return shift

def gaussian_derivative(coordinates, image_center, alpha, mu, sigma, dimension, rotation=0):
    """
    Computes the derivative of a Gaussian function in dimension 'dimension'.
    """

    mu_normalized = mu + image_center

    if rotation != 0:
        coordinates = transform_coordinates(coordinates, rotation, mu_normalized)


    shift = gaussian(coordinates, mu_normalized, sigma, alpha)
    shift = shift * (coordinates[dimension] - mu_normalized[dimension]) / sigma[dimension] ** 2

    return shift

def test_gaussian_shift():
    """
    Tests the gaussian_shift function.
    :return:
    """
    coordinates = np. meshgrid(np.arange(10), np.arange(10))

def apply_gaussian_lensing(image, alpha_dirs, mu_dirs, sigma_dirs, rotation_dirs, **kwargs):
    """
    Applies a Gaussian transform to an image.
    """

    if not all([image.ndim == mu_dirs[i].shape[0] for i in range(len(alpha_dirs))]):
        raise ValueError("Image and all directions must have the same first dimension size.")

    image_center = np.array([np.floor(image.shape[i] / 2) for i in range(len(image.shape))])
    shape = image.shape

    coordinates = np.stack(np.indices(shape))
    vector_field = []

    for dimension, (alpha, mu, sigma, rotation)in enumerate(zip(alpha_dirs, mu_dirs, sigma_dirs, rotation_dirs)):

        vector_field += [gaussian_shift(coordinates=coordinates,
                                        image_center=image_center,
                                        alpha=alpha,
                                        mu=mu,
                                        sigma=sigma,
                                        dimension=dimension,
                                        rotation=rotation,
                                        **kwargs)]

    vector_field = np.stack(vector_field, axis=0)
    coordinates_transformed = vector_field + coordinates

    # Apply the Gaussian vector field to the image
    transformed_image = map_coordinates(image, coordinates_transformed, order=1)

    return transformed_image, vector_field



def apply_gaussian_transform(image, alpha_dirs, mu_dirs, sigma_dirs, rotation_dirs, **kwargs):
    """
    Applies a Gaussian transform to an image.
    """

    if not all([image.ndim == mu_dirs[i].shape[0] for i in range(len(alpha_dirs))]):
        raise ValueError("Image and all directions must have the same first dimension size.")

    image_center = np.array([np.floor(image.shape[i] / 2) for i in range(len(image.shape))])
    shape = image.shape

    coordinates = np.stack(np.indices(shape))
    vector_field = []

    for dimension, (alpha, mu, sigma, rotation)in enumerate(zip(alpha_dirs, mu_dirs, sigma_dirs, rotation_dirs)):

        vector_field += [gaussian_derivative(coordinates=coordinates,
                                        image_center=image_center,
                                        alpha=alpha,
                                        mu=mu,
                                        sigma=sigma,
                                        dimension=dimension,
                                        rotation=rotation,
                                        **kwargs)]

    vector_field = np.stack(vector_field, axis=0)
    coordinates_transformed = vector_field + coordinates

    # Apply the Gaussian vector field to the image
    transformed_image = map_coordinates(image, coordinates_transformed, order=1)

    return transformed_image, vector_field

def visualize_vector_field(vector_field):
    import matplotlib.pyplot as plt

    plt.quiver(vector_field[1, :, :], vector_field[0, :, :])
    plt.show()

def load_test_image():
    from phantom_helpers.binary_tools import read_binary, compare_images
    try:
        BASE_PATH = "/home/j/J.Titze/Projects/XCAT_data/Phantoms/"
        postfix = "_atn_1.bin"

        path_original = "high"

        image_dimenstions = (512, 512, 40)
        slice = 20

        image_original = read_binary(os.path.join(BASE_PATH, path_original + postfix), *image_dimenstions)[slice, :, :]

    except FileNotFoundError:
        warnings.warn("Could not find test image.\n "
                      "Please download the XCAT phantom and set the correct path.")
        image_original = np.ones((20, 20))

    return image_original

def transform_coordinates(coordinates, rotation_angle, mu_normalized):
    """
    rotate coordinates around mu by angle rotation_angle
    """
    rotation_angle_rad = np.radians(rotation_angle)

    rotation_matrix = np.array([[np.cos(rotation_angle_rad), -np.sin(rotation_angle_rad)],
                                [np.sin(rotation_angle_rad), np.cos(rotation_angle_rad)]])

    coordinates_shifted = coordinates - mu_normalized[:,None,None]
    coordinates_rotated = np.einsum("ij, jlm -> ilm", rotation_matrix, coordinates_shifted)
    coordinates_transformed = coordinates_rotated + mu_normalized[:,None,None]

    return coordinates_transformed


def load_dummy_image():
    image_original = np.ones((20, 20))
    return image_original

#make gaussian parameters a named tuple


def test_apply_gaussian_transform(gaussian_parameters=None):
    """
    Tests the apply_gaussian_transform function with plots.
    """
    from phantom_helpers.binary_tools import compare_images

    if gaussian_parameters is None:
        gaussian_parameters = {
            "alpha_dirs": [0.5, 0.5],
            "mu_dirs": np.array([[3, 3],
                                 [3, 3]]),
            "sigma_dirs": [np.array([5, 5]),
                           np.array([5, 5])],
            "rotation_dirs": [0, 0],
        }

    image_original = load_test_image()

    image_warped, vector_field = apply_gaussian_transform(image_original, **gaussian_parameters)

    visualize_vector_field(vector_field)

    compare_images(image_original, image_warped)

def compare_gaussian_transforms():
    from phantom_helpers.binary_tools import compare_images

    gaussian_parameters = {
        "alpha_dirs": [1, 0],
        "mu_dirs": np.array([[0, 0],
                             [0, 0]]),
        "sigma_dirs": [np.array([4, 4]),
                       np.array([4, 4])],
        "rotation_dirs": [0, 0],
    }

    image_original = load_dummy_image()

    image_warped, vector_field = apply_gaussian_transform(image_original, **gaussian_parameters)

    image_lensing, vector_lensing = apply_gaussian_lensing(image_original, **gaussian_parameters, lensing=True)

    image_shifted, vector_shifted = apply_gaussian_lensing(image_original, **gaussian_parameters, lensing=False)

    visualize_vector_field(vector_field)

    visualize_vector_field(vector_lensing)
    visualize_vector_field(vector_shifted)

    #compare_images(image_original, image_warped)

def test_sample_gaussian_transform():
    from phantom_helpers.binary_tools import compare_images

    sample_parameters = {
    }

    gaussian_parameters = sample_gaussian_parameters(**sample_parameters)
    print(gaussian_parameters)

    image_original = load_test_image()
    image_warped = apply_gaussian_transform(image_original, **gaussian_parameters)
    compare_images(image_original, image_warped)



if __name__ == '__main__':
    gaussian_parameters = {
        "alpha_dirs": [50, 50],
        "mu_dirs": np.array([[0, 0],
                             [0, 0]]),
        "sigma_dirs": [np.array([100, 50]),
                       np.array([50, 100])],
        "rotation_dirs": [0, 0],
    }
    test_apply_gaussian_transform(gaussian_parameters=gaussian_parameters)
    #compare_gaussian_transforms()
    #test_sample_gaussian_transform()

    #TODO make visualiuzation with grey underlay original