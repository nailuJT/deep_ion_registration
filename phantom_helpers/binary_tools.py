import numpy as np
import struct
import matplotlib.pyplot as plt
import os

from augmentation.helpers.plotting import compare_images


def read_binary(path, shape_x, shape_y, slices):
    """
    Reads a binary file and returns a numpy array with shape (slices, shape_x, shape_y).

    :param path: The path to the binary file.
    :param shape_x: The size of the x dimension of the output array.
    :param shape_y: The size of the y dimension of the output array.
    :param slices: The number of slices in the output array.
    :return: A numpy array with the unpacked binary data.
    """
    size_slice = shape_x * shape_y
    binary_format = '<' + str(size_slice * slices) + 'f'

    with open(path, 'rb') as f:
        binary = f.read()
        image_flatt = struct.unpack(binary_format, binary)
        image_array = np.reshape(image_flatt, (slices, shape_x, shape_y))

    return image_array

# def compare_images(original_image, warped_image, plot=True, ax=None):
#     """
#     Compares two images by plotting the difference between them.
#
#     :param original_image:
#     :param warped_image:
#     :param plot:
#     :param ax:
#     :return:
#     """
#     diff_image = np.abs(original_image - warped_image)
#
#     if plot:
#         if ax is None:
#             fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#
#         ax[0].imshow(original_image)
#         ax[1].imshow(warped_image)
#         ax[2].imshow(diff_image)
#         plt.show()
#
#     return diff_image

def test_compare_images():
    path = "/home/j/J.Titze/Projects/XCAT_data/Phantoms/test3_atn_1.bin"
    data = read_binary(path, 256, 256, 150)
    data = data[20,:,:]
    compare_images(data, data)


def test_read_binary():
    path = "/home/j/J.Titze/Projects/XCAT_data/Phantoms/test3_atn_1.bin"
    data = read_binary(path, 256, 256, 150)
    print(data.shape)
    plt.imshow(data[20,:,:])
    plt.show()


def main():
    BASE_PATH = "/home/j/J.Titze/Projects/XCAT_data/Phantoms/"
    path_original = os.path.join(BASE_PATH, "test_baseline_atn_1.bin")
    path_warped = os.path.join(BASE_PATH, "test_skin_atn_1.bin")

    image_original = read_binary(path_original, 256, 256, 150)
    image_warped = read_binary(path_warped, 256, 256, 150)

    compare_images(image_original[80, :, :], image_warped[80, :, :])

if __name__ == '__main__':
    main()


