import numpy as np
import os
from scipy.ndimage import map_coordinates

def gaussian(x, mu, sig):
    y = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    return y


def sample_gaussian_parameters(center_mu,
                               center_sigma,
                               epsilon_mu,
                               epsilon_sigma,
                               epsilon_max,
                               epsilon_min,
                               sigma_mu,
                               sigma_sigma,
                               sigma_max,
                               sigma_min,
                               lensing_chance=0.5,
                               dim=2):

    epsilon = np.random.normal(epsilon_mu, epsilon_sigma, dim)
    epsilon = np.clip(epsilon, epsilon_min, epsilon_max)

    sigma = np.random.normal(sigma_mu, sigma_sigma, dim)
    sigma = np.clip(sigma, sigma_min, sigma_max)

    mu = np.random.normal(center_mu, center_sigma, dim)

    lensing = np.random.choice([True, False], dim, p=[lensing_chance, 1 - lensing_chance])

    parameters = {
        "epsilon": epsilon,
        "mu": mu,
        "sigma": sigma
    }

    return parameters



def apply_gaussian_transform(image, epsilon, mu, sigma, lensing=True):
    """
    Applies a Gaussian transform to an image.
    """
    center = (np.floor(image.shape[0] / 2), np.floor(image.shape[1] / 2))

    # Create a Gaussian vector field
    y_indices, x_indices = np.indices(image.shape)

    sign = [1, 1]
    lensing = True

    if lensing:
        sign[0] = np.where(x_indices < center[1], -1, 1)
        sign[1] = np.where(y_indices < center[0], -1, 1)

    vector_field = np.stack([
        sign[0] * epsilon[0] * gaussian(x_indices, mu[0] + center[0], sigma[0]),
        sign[1] * epsilon[1] * gaussian(y_indices, mu[1] + center[1], sigma[1])
    ], axis=0)

    coordinates = np.stack([y_indices, x_indices], axis=0)
    vector_field = vector_field + coordinates

    # Apply the Gaussian vector field to the image
    transformed_image = map_coordinates(image, vector_field, order=1)

    return transformed_image

def load_test_image():
    from phantom_helpers.binary_tools import read_binary, compare_images

    BASE_PATH = "/home/j/J.Titze/Projects/XCAT_data/Phantoms/"
    postfix = "_atn_1.bin"

    path_original = "high"

    image_dimenstions = (512, 512, 40)
    slice = 20

    image_original = read_binary(os.path.join(BASE_PATH, path_original + postfix), *image_dimenstions)[slice,:,:]

    return image_original

def test_apply_gaussian_transform():
    """
    Tests the apply_gaussian_transform function with plots.
    """
    from phantom_helpers.binary_tools import compare_images

    gaussian_parameters = {
        "epsilon": np.array([5, 5]),
        "mu": np.array([20, 0]),
        "sigma": np.array([10, 10])
    }

    image_original = load_test_image()
    image_warped = apply_gaussian_transform(image_original, **gaussian_parameters)
    compare_images(image_original, image_warped)

def test_sample_gaussian_transform():
    from phantom_helpers.binary_tools import compare_images

    sample_parameters = {
        "center_mu": 0,
        "center_sigma": 50,
        "epsilon_mu": 5,
        "epsilon_sigma": 5,
        "epsilon_max": 10,
        "epsilon_min": 3,
        "sigma_mu": 10,
        "sigma_sigma": 5,
        "sigma_max": 10,
        "sigma_min": 0,
        "dim": 2
    }

    gaussian_parameters = sample_gaussian_parameters(**sample_parameters)
    print(gaussian_parameters)

    image_original = load_test_image()
    image_warped = apply_gaussian_transform(image_original, **gaussian_parameters)
    compare_images(image_original, image_warped)



if __name__ == '__main__':
    #test_apply_gaussian_transform()
    test_sample_gaussian_transform()