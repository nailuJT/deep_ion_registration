import numpy as np
from matplotlib import pyplot as plt


def compare_images(original_image, warped_image, plot=True, ax=None):
    """
    Compares two images by plotting the difference between them.

    :param original_image:
    :param warped_image:
    :param plot:
    :param ax:
    :return:
    """
    from matplotlib.colors import LinearSegmentedColormap

    diff_image = np.abs(original_image - warped_image)

    if plot:
        if ax is None:
            fig, ax = plt.subplots(3, 1, figsize=(6, 15))

        ax[0].imshow(original_image)
        ax[1].imshow(warped_image)

        # Plot the warped image in grayscale
        ax[2].imshow(original_image, cmap='gray')

        # Create a colormap that goes from transparent to red
        colors = [(0, 0, 0, 0),(0, 0, 1, 0.7),(0, 1, 0, 1), (1, 0, 0, 1)]  # R -> G -> B -> A
        cmap_name = 'my_list'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

        # Plot the difference image with the new colormap
        ax[2].imshow(diff_image, cmap=cm)

        plt.show()

    return diff_image


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


def plot_projections(projection):
    """
    Plots the projections.
    """
    plt.figure()
    plt.imshow(projection)
    plt.show()


def plot_comparison(projection, projection_ines):
    """
    Plots the projections.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(projection)
    axs[1].imshow(projection_ines)
    plt.show()


def plot_slice(slice):
    """
    Plots a slice.
    """
    plt.figure()
    plt.imshow(slice)
    plt.show()
