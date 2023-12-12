import numpy as np
import os
from scipy.ndimage import map_coordinates
import warnings
from collections import namedtuple
from
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