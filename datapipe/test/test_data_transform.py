from datapipe.data_transform import apply_gaussian_transform3d, GaussianParameters
from datapipe.deformation_sampling import GaussianParameterSampler, transform_projection
from datapipe.generate_data_straight_rework import PatientCT, Projection
import numpy as np
import matplotlib.pyplot as plt
import cProfile

def load_projection():
    patient_name = "male1"
    patient = PatientCT(patient_name)
    angles = np.linspace(0, 180, 1, endpoint=False)
    projection = Projection(patient, angles)
    return projection

def test_projection_transform():

    projection = load_projection()
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


    projection_transformed, vector_field = transform_projection(projection, gaussian_parameters)

    plt.imshow(projection_transformed.patient.ct[20,:,:])
    plt.show()


    angles = projection_transformed.generate()

    plt.imshow(angles[0][20,:,:])
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
    projection_transformed, vector_field = transform_projection(projection, gaussian_parameters)

    # Stop profiling
    profiler.disable()

    # Print the profiling results
    profiler.print_stats()

    plt.imshow(projection_transformed.patient.ct[20,:,:])
    plt.show()

def test_gaussian_sampling():
    sampler =
    print(sampler.sample())



if __name__ == '__main__':
    test_projection_transform()