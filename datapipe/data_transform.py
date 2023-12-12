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