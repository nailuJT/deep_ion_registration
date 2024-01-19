import numpy as np
import os
from scipy.ndimage import map_coordinates
import warnings
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R


@dataclass
class GaussianParameters:
    alpha_dirs: any
    mu_dirs: any
    sigma_dirs: any
    rotation_dirs: any

def gaussian3d(x, mu, sigma, epsilon=1e-8):
    """
    Computes a Gaussian function.
    """

    dist = np.power(x - mu[:, None, None, None], 2)
    dist = np.divide(dist, 2 * sigma[:, None, None, None] ** 2 + epsilon)
    dist = np.sum(dist, axis=0)
    y = np.exp(-dist)

    return y



def gaussian_derivative_3d(coordinates, alpha, mu, sigma, dimension, normalize=True, **_):
    """
    Computes the derivative of a Gaussian function in dimension 'dimension'.
    """
    shift = gaussian3d(coordinates, mu, sigma, alpha)
    shift = alpha * shift * (coordinates[dimension] - mu[dimension])

    if normalize:
        shift = shift / sigma[dimension] ** 2

    return shift


def apply_gaussian_transform3d(image, alpha_dirs, mu_dirs, sigma_dirs, rotation_dirs, **kwargs):
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

        mu_normalized = mu + image_center

        coordinates_rotated = rotate_coordinates_3d(coordinates, rotation, mu_normalized)

        coordinates_rotated = coordinates

        vector_field += [gaussian_derivative_3d(coordinates=coordinates_rotated,
                                                alpha=alpha,
                                                mu=mu_normalized,
                                                sigma=sigma,
                                                dimension=dimension,
                                                **kwargs)]

    vector_field = np.stack(vector_field, axis=0)
    coordinates_transformed = vector_field + coordinates

    # Apply the Gaussian vector field to the image
    transformed_image = map_coordinates(image, coordinates_transformed, order=1)

    return transformed_image, vector_field



def rotate_coordinates_3d(coordinates, rotation_angles, mu):
    """
    Rotate 3D coordinates around mu by rotation_angles
    """
    # Convert rotation angles to radians
    rotation_angles_rad = np.radians(rotation_angles)

    # Create a rotation object
    rotation = R.from_euler('xyz', rotation_angles_rad)
    rotation_matrix = rotation.as_matrix()

    coordinates_shifted = coordinates - mu[:, None, None, None]
    coordinates_rotated = np.tensordot(rotation_matrix, coordinates_shifted, axes=((1), (0)))
    coordinates_transformed = coordinates_rotated + mu[:, None, None, None]

    return coordinates_transformed

def gaussian2d(x, mu, sigma, alpha, epsilon=1e-8):
    """
    Computes a Gaussian function.
    """

    dist = np.power(x - mu[:, None, None], 2)
    dist = np.divide(dist, 2 * sigma[:, None, None] ** 2 + epsilon)
    dist = np.sum(dist, axis=0)
    y = alpha * np.exp(-dist)

    return y



def gaussian_derivative_2d(coordinates, alpha, mu, sigma, dimension, normalize=True, **_):
    """
    Computes the derivative of a Gaussian function in dimension 'dimension'.
    """
    shift = gaussian2d(coordinates, mu, sigma, alpha)
    shift = shift * (coordinates[dimension] - mu[dimension])

    if normalize:
        shift = shift / sigma[dimension] ** 2

    return shift

def transform_coordinates2d(coordinates, rotation_angle, mu_normalized):
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


def apply_gaussian_transform2d(image, alpha_dirs, mu_dirs, sigma_dirs, rotation_dirs, **kwargs):
    """
    Applies a Gaussian transform to an image.
    """
    if not all([image.ndim == mu_dirs[i].shape[0] for i in range(len(alpha_dirs))]):
        raise ValueError("Image and all directions must have the same first dimension size.")

    image_center = np.array([np.floor(image.shape[i] / 2) for i in range(len(image.shape))])
    shape = image.shape

    coordinates = np.stack(np.indices(shape))
    vector_field = []

    for dimension, (alpha, mu, sigma, rotation) in enumerate(zip(alpha_dirs, mu_dirs, sigma_dirs, rotation_dirs)):

        mu_normalized = mu + image_center

        coordinates_rotated = transform_coordinates2d(coordinates, rotation, mu_normalized)

        vector_field += [gaussian_derivative_2d(coordinates=coordinates_rotated,
                                                alpha=alpha,
                                                mu=mu_normalized,
                                                sigma=sigma,
                                                dimension=dimension,
                                                **kwargs)]

    vector_field = np.stack(vector_field, axis=0)
    coordinates_transformed = vector_field + coordinates

    # Apply the Gaussian vector field to the image
    transformed_image = map_coordinates(image, coordinates_transformed, order=1)

    return transformed_image, vector_field



if __name__ == '__main__':
    gaussian_parameters = {
        "alpha_dirs": [50, 50],
        "mu_dirs": np.array([[0, 0],
                             [0, 0]]),
        "sigma_dirs": [np.array([100, 50]),
                       np.array([50, 100])],
        "rotation_dirs": [0, 0],
    }
    #test_apply_gaussian_transform(gaussian_parameters=gaussian_parameters)
    #compare_gaussian_transforms()
    #test_sample_gaussian_transform()

    #TODO make visualiuzation with grey underlay original