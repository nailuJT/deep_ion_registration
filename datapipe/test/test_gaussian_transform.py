import numpy as np
import os
from scipy.ndimage import map_coordinates
import warnings
from collections import namedtuple
from datapipe.data_transform import apply_gaussian_transform3d, apply_gaussian_transform2d


def visualize_vector_field(vector_field):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.quiver(vector_field[1, :, :], vector_field[0, :, :])
    plt.show()

def visualize_vector_field_big(vector_field, num_samples=30):
    import matplotlib.pyplot as plt
    # Compute the step size for each dimension
    step_size = np.maximum(np.array(vector_field.shape[1:]) // num_samples, 1)

    # Subsample the vector field
    subsampled_vector_field = vector_field[:, ::step_size[0], ::step_size[1]]/step_size[0]


    fig, ax = plt.subplots(3, 1, figsize=(6, 15))
    # plot with titles and labels
    ax[0].imshow(vector_field[0, :, :], origin='lower')
    ax[0].set_title('Vector Field - X Component')

    ax[1].imshow(vector_field[1, :, :], origin='lower')
    ax[1].set_title('Vector Field - Y Component')

    ax[2].quiver(subsampled_vector_field[1, :, :], subsampled_vector_field[0, :, :])
    ax[2].set_title('Subsampled Vector Field')

def visualize_vector_field_3d(vector_field):
    import matplotlib.pyplot as plt
    fig = plt.figure( figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')


    vector_field = vector_field.transpose(0, 2, 3, 1)

    x, y, z = np.meshgrid(np.arange(vector_field.shape[1]),
                          np.arange(vector_field.shape[2]),
                          np.arange(vector_field.shape[3]))

    ax.quiver(x, y, z, vector_field[0, :, :, :], vector_field[1, :, :, :], vector_field[2, :, :, :])

def visualize_vector_field_with_timeout(vector_field, timeout):
    import multiprocessing

    p = multiprocessing.Process(target=visualize_vector_field, args=(vector_field,))
    p.start()
    p.join(timeout)

    if p.is_alive():
        print("The function 'visualize_vector_field' took too long to complete. It has been terminated.")
        p.terminate()
        p.join()

    else:
        visualize_vector_field_big(vector_field)

    plt.show()
def load_test_image():
    from phantom_helpers.binary_tools import read_binary, compare_images
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
    from phantom_helpers.binary_tools import read_binary, compare_images
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
    from phantom_helpers.binary_tools import compare_images

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