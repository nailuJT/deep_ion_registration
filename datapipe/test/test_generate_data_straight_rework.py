
from datapipe.generate_data_straight import generate_sysm
from datapipe.generate_data_straight_rework import generate_system_matrix, PatientCT, generate_projections
import numpy as np
import matplotlib.pyplot as plt


def compare_generate_system_matrix():
    """
    Tests the generate_system_matrix function.
    """
    n_angles = 10
    patient = PatientCT('male1')
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    system_matrices_angles = generate_system_matrix(patient.slice_shape, angles)
    system_matrices_angles_ines = generate_sysm(n_angles, return_sys_angles=True)

    for i, theta in enumerate(angles):
        assert np.allclose(system_matrices_angles[theta].todense(), system_matrices_angles_ines[i].todense())

def compare_generate_projections():
    """
    Tests the generate_projections function.
    """
    patient = PatientCT('male1')
    n_angles = 10
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    system_matrices_angles = generate_system_matrix(patient.slice_shape, angles)
    projections = generate_projections(patient, system_matrices_angles, normalize=True)

    _ , projections_ines, _ = generate_sysm(n_angles,
                                         force_patients=['male1'],
                                         return_proj_angle=True,
                                         stop_reorder=True)

    projection_ines_angles = {}
    # reshape to match the shape of the projections
    for i, theta in enumerate(angles):
        projections_ines_angle = np.zeros((patient.shape[0], patient.shape[1]))
        for k in range(patient.n_slices):

            projection_ines_angele_slice = projections_ines[n_angles*k + i].flatten()
            projections_ines_angle[k] = projection_ines_angele_slice


        projection_ines_angles[theta] = projections_ines_angle

    for i, theta in enumerate(angles):
        plot_comparison(projections[theta], projection_ines_angles[theta])

    for i, theta in enumerate(angles):
        assert np.allclose(projections[theta], projection_ines_angles[theta])

def compare_ion_cts():
    """
    Tests the ion_ct property of the PatientCT class.
    """
    patient = PatientCT('male1')
    ion_cts = patient.ion_ct
    _, _, ion_cts_ines = generate_sysm(1, force_patients=['male1'], stop_reorder=True)
    ion_cts_ines = np.stack(ion_cts_ines).squeeze()
    for i in range(patient.n_slices):
        plot_comparison(ion_cts[i], ion_cts_ines[i])
        assert np.allclose(ion_cts[i], ion_cts_ines[i])



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


def test_ion_ct():
    """
    Tests the ion_ct property of the PatientCT class.
    """
    patient = PatientCT('male1')
    print(patient.ion_ct.shape)

def test_generate_system_matrix():
    """
    Tests the generate_system_matrix function.
    """
    patient = PatientCT('male1')
    angles = np.linspace(0, 180, 10, endpoint=False)
    system_matrices_angles = generate_system_matrix(patient.slice_shape, angles)
    print(system_matrices_angles[0].shape)


def test_generate_projections():
    """
    Tests the generate_projections function.
    """
    save_path = '/home/j/J.Titze/Data/system_matrices'

    patient = PatientCT('male1')
    angles = np.linspace(0, 180, 10, endpoint=False)
    system_matrices_angles = generate_system_matrix(patient.shape, angles)
    generate_projections(patient, system_matrices_angles,
                         save_path=save_path,
                         n_ions=1,
                         normalize=True,
                         save=False,
                         slice_block=1)


if __name__ == '__main__':
    #compare_generate_system_matrix()
    compare_generate_projections()
    compare_ion_cts()

