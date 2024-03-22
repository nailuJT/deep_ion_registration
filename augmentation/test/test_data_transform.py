from datapipe.data_transform import apply_gaussian_transform3d, GaussianParameters
from datapipe.deformation_sampling import GaussianParameterSampler, transform_ct
from datapipe.straight_projection import PatientCT, Projector
import numpy as np
import matplotlib.pyplot as plt
import cProfile
from phantom_helpers.binary_tools import compare_images

def load_projection():
    patient_name = "male1"
    patient = PatientCT(patient_name)
    angles = np.linspace(0, 180, 1, endpoint=False)
    projection = Projector(patient.slice_shape, angles)
    return projection, patient

def test_projection_transform():

    projection, patient = load_projection()
    gaussian_parameters = GaussianParameters(alpha_dirs=np.array([1000, 1000, 1000]),
                                                mu_dirs=np.array([[100, 0, 0],
                                                                [100, 0, 0],
                                                                [100, 0, 0]]),
                                                sigma_dirs=np.array([[40, 40, 40],
                                                                    [40, 40, 40],
                                                                    [40, 40, 40]]),
                                                rotation_dirs=np.array([[0, 0, 0],
                                                                        [0, 0, 0],
                                                                        [0, 0, 0]]))

    patient_transformed, vector_field = transform_ct(patient, gaussian_parameters.__dict__)

    plt.imshow(patient_transformed.ct[20,:,:])
    plt.show()

    compare_images(patient.ct[20,:,:], patient_transformed.ct[20,:,:])


    angles = projection.generate(patient_transformed)

    plt.imshow(angles[0])
    plt.show()


def test_sampled_transform():


    projection, patient = load_projection()

    config_dict = {
        "alpha_mean": [300, 1500, 1500],
        "alpha_std": [400, 2000, 2000],
        "mu_mean": [[0.0, 0.0, 12.0], [10.0, 0.0, 12.0], [0.0, 0.0, 12.0]],
        "mu_std": [[40.0, 40.0, 5.0], [40.0, 40.0, 5.0], [40.0, 40.0, 5.0]],
        "sigma_mean": [[25, 25, 12], [25, 25, 12], [25, 25, 12]],
        "sigma_std": [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
        "rotation_mean": [0.0, 0.0, 0.0],
        "rotation_std": [40.0, 40.0, 40.0]
    }

    sampler = GaussianParameterSampler.from_dict(config_dict)

    gaussian_parameters = sampler.sample(inversion="none")

    patient_transformed, vector_field = transform_ct(patient, gaussian_parameters)


    plt.imshow(patient_transformed.ct[20,:,:])
    plt.show()

    compare_images(patient.ct[20,:,:], patient_transformed.ct[20,:,:])


    angles = projection.generate(patient_transformed)

    plt.imshow(angles[0])
    plt.show()

def profile_projection_transform():

    projection = load_projection()
    gaussian_parameters = GaussianParameters(alpha_dirs=np.array([0, np.pi/2, np.pi]),
                                                mu_dirs=np.array([[0, 0, 0],
                                                               [0, 0, 0],
                                                                [0, 0, 0]]),
                                                sigma_dirs=np.array([[1, 1, 1],
                                                                    [1, 1, 1],
                                                                    [1, 1, 1]]),
                                                rotation_dirs=np.array([[0, 0, 0],
                                                                        [0, 0, 0],
                                                                        [0, 0, 0]]))

    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()

    # Call the method you want to profile
    projection_transformed, vector_field = transform_ct(projection, gaussian_parameters)

    # Stop profiling
    profiler.disable()

    # Print the profiling results
    profiler.print_stats()

    plt.imshow(projection_transformed.patient.ct[20,:,:])
    plt.show()

def test_gaussian_sampling():
    sampler = GaussianParameterSampler.from_config({
        "alpha_mean": [300, 1500, 1500],
        "alpha_std": [400, 2000, 2000],
        "mu_mean": [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        "mu_std": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        "sigma_mean": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "sigma_std": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        "rotation_mean": [0.0, 0.0, 0.0],
        "rotation_std": [1.0, 1.0, 1.0]
    })
    print(sampler.sample())

if __name__ == '__main__':
    test_projection_transform()
    # test_sampled_transform()