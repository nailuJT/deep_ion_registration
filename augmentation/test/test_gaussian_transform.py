import numpy as np
import os
import warnings
from augmentation.data_transform import apply_gaussian_transform3d, apply_gaussian_transform2d
from augmentation.helpers.plotting import visualize_vector_field_big


def load_test_image():
    from phantom_helpers.binary_tools import read_binary
    try:
        BASE_PATH = "/home/j/J.Titze/Projects/XCAT_data/Phantoms/"
        postfix = "_atn_1.bin"

        path_original = "high"

        image_dimensions = (512, 512, 40)
        slice = 20

        image_original = read_binary(os.path.join(BASE_PATH, path_original + postfix), *image_dimensions)[slice, :, :]

    except FileNotFoundError:
        warnings.warn("Could not find test image.\n "
                      "Please download the XCAT phantom and set the correct path.")
        image_original = np.ones((20, 20))

    return image_original

def load_test_image_3d():
    from phantom_helpers.binary_tools import read_binary
    try:
        BASE_PATH = "/home/j/J.Titze/Projects/XCAT_data/Phantoms/"
        postfix = "_atn_1.bin"

        path_original = "high"

        image_dimenstions = (512, 512, 40)

        image_original = read_binary(os.path.join(BASE_PATH, path_original + postfix), *image_dimenstions)

    except FileNotFoundError:
        warnings.warn("Could not find test image.\n "
                      "Please download the XCAT phantom and set the correct path.")
        image_original = np.ones((20, 20, 20))

    return image_original


def load_dummy_image():
    image_original = np.zeros((20, 20))

    #draw a circle
    for i in range(20):
        for j in range(20):
            if (i-10)**2 + (j-10)**2 < 7**2:
                image_original[i, j] = 1

    return image_original

def load_dummy_image_3d():
    image_original = np.ones((15, 15, 15))

    #draw a circle
    for i in range(15):
        for j in range(15):
            for k in range(15):
                if (i-7)**2 + (j-7)**2 + (k-7)**2 < 5**2:
                    image_original[i, j, k] = 0
    return image_original

#make gaussian parameters a named tuple

def test_apply_gaussian_transform3d(gaussian_parameters=None, dummy=False):
    """
    Tests the apply_gaussian_transform function with plots.
    """
    from phantom_helpers.binary_tools import compare_images

    if gaussian_parameters is None:
        gaussian_parameters = {
            "alpha_dirs": [-40, -40, 0],
            "mu_dirs": np.array([[100, 3, 0],
                                 [100, 3, 0],
                                 [0, 0, 0]]),
            "sigma_dirs": [np.array([50, 50, 100]),
                            np.array([50, 50, 100]),
                            np.array([50, 50, 100])],
            "rotation_dirs": [[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]],
        }

    if dummy:
        image_original = load_dummy_image_3d()

    else:
        image_original = load_test_image_3d()

    image_original = image_original.transpose(1, 2, 0)

    image_warped, vector_field = apply_gaussian_transform3d(image_original, **gaussian_parameters)

    #visualize_vector_field_3d(vector_field)

    compare_images(image_original[:,:,20], image_warped[:,:, 20])

    return image_original, image_warped, vector_field


def test_apply_gaussian_transform2d(gaussian_parameters=None, dummy=False):
    """
    Tests the apply_gaussian_transform function with plots.
    """
    from phantom_helpers.binary_tools import compare_images

    if gaussian_parameters is None:
        gaussian_parameters = {
            "alpha_dirs": [-100, -100],
            "mu_dirs": np.array([[100, 3],
                                 [-100, 3]]),
            "sigma_dirs": [np.array([50, 50]),
                           np.array([50, 50])],
            "rotation_dirs": [0, 0],
        }

    if dummy:
        image_original = load_dummy_image()

    else:
        image_original = load_test_image()

    image_warped, vector_field = apply_gaussian_transform2d(image_original, **gaussian_parameters)

    # try visualize vector field and timeout if not possible
    visualize_vector_field_big(vector_field, 50)

    compare_images(image_original, image_warped)
    #compare_images(image_original, image_warped)

    return image_original, image_warped, vector_field

def test_sample_gaussian_transform():
    from augmentation.helpers.plotting import compare_images

    sample_parameters = {
    }

    gaussian_parameters = sample_gaussian_parameters(**sample_parameters)
    print(gaussian_parameters)

    image_original = load_test_image()
    image_warped = apply_gaussian_transform3d(image_original, **gaussian_parameters)
    compare_images(image_original, image_warped)

if __name__ == '__main__':
    gaussian_parameters2d = {
        "alpha_dirs": [-100, -100],
        "mu_dirs": np.array([[100, 3],
                             [100, 3]]),
        "sigma_dirs": [np.array([50, 50]),
                       np.array([50, 50])],
        "rotation_dirs": [0, 0],
    }

    gaussian_parameters = {
        "alpha_dirs": [-100, -100, 0],
        "mu_dirs": np.array([[100, 3, 0],
                             [100, 3, 0],
                             [0, 0, 0]]),
        "sigma_dirs": [np.array([50, 50, 100]),
                        np.array([50, 50, 100]),
                        np.array([50, 50, 100])],
        "rotation_dirs": [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],
    }


    _, _, vector_field2d = test_apply_gaussian_transform2d(gaussian_parameters=gaussian_parameters2d,)

    # for alpha in range(-1000, 0, 100):
    #
    #     gaussian_parameters = {
    #         "alpha_dirs": [alpha, alpha, 0],
    #         "mu_dirs": np.array([[100, 3, 0],
    #                              [100, 3, 0],
    #                              [0, 0, 0]]),
    #         "sigma_dirs": [np.array([50, 50, 100]),
    #                        np.array([50, 50, 100]),
    #                        np.array([50, 50, 100])],
    #         "rotation_dirs": [[0, 0, 0],
    #                           [0, 0, 0],
    #                           [0, 0, 0]],
    #     }
    #
    #     _, _, vector_field = test_apply_gaussian_transform3d(gaussian_parameters=gaussian_parameters, dummy=False)
    #
    #     print(vector_field.max())



    #compare_gaussian_transforms()
    #test_sample_gaussian_transform()