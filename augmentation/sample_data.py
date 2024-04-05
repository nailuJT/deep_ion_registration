import matplotlib.pyplot as plt
import numpy as np
import os
from warnings import warn
from augmentation.deformation_sampling import GaussianParameterSampler, transform_ct
from augmentation.straight_projection import PatientCT, Projector
from tqdm import tqdm

SLICES = 76
DATA_DIR = "/project/med6/IONCT/julian_titze/data/raw"
if not os.path.exists(DATA_DIR):
    warn(f"Folder {DATA_DIR} does not exist")

def sample_data():

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
    slices_centers = [-30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    mu_means = [[[center, 0., 0.], [center, 0., 0.], [center, 0., 0.]] for center in slices_centers]
    configs = [config_dict.copy() for _ in range(len(slices_centers))]
    for i, mu_mean in enumerate(mu_means):
        configs[i]["mu_mean"] = mu_mean

    samplers = [GaussianParameterSampler.from_dict(config) for config in configs]

    patient_names = ["male1", "male2", "female1", "female2", "female3", "female4"]
    patients = [PatientCT(name) for name in patient_names]

    n_angles = 5
    angles = np.linspace(0, 180, n_angles)

    projector = Projector(angles=angles, slice_shape=(patients[0].slice_shape))

    for patient_index, patient in enumerate(patients):
        print(f"Processing patient {patient_index + 1} of {len(patients)}")
        for sampler_index, sampler in enumerate(samplers):
            print(f"\tDeformation {sampler_index + 1} of {len(samplers)}")
            transformed_patient, vector_field = transform_ct(patient, sampler.sample(inversion="none"))

        projection_angles = projector.generate(transformed_patient)
        # angels dict to array
        projection_angles = np.array([val for val in projection_angles.values()])

        # save patient.ionct, patient.mask, angles and vector_field
        chunk_size = 8
        offset = 3

        # Number of chunks
        n_chunks = patient.n_slices // chunk_size

        for i in range(n_chunks):
            print(f"\t\tProcessing chunk {i + 1} of {n_chunks}")
            # Get the start and end index for the current chunk
            start = i * chunk_size + offset
            end = start + chunk_size + offset

            # Get the chunk data
            ionct_chunk = patient.ion_ct[start:end, :, :]
            patient_ct = patient.ct[start:end, :, :]
            transformed_ionct_chunk = transformed_patient.ion_ct[start:end, :, :]
            mask_chunk = patient.mask[start:end, :, :]
            projection_angles_chunk = projection_angles[:, start:end, :]
            vector_field_chunk = vector_field[start:end]

            # Save the chunk data to a numpy file
            np.save(os.path.join(DATA_DIR, f"ionct_chunk_{patient.name}_{i}.npy"), ionct_chunk)
            np.save(os.path.join(DATA_DIR, f"ct_chunk_{patient.name}_{i}.npy"), patient_ct)
            np.save(os.path.join(DATA_DIR, f"transformed_ionct_chunk_{patient.name}_{i}.npy"), transformed_ionct_chunk)
            np.save(os.path.join(DATA_DIR, f"mask_chunk_{patient.name}_{i}.npy"), mask_chunk)
            np.save(os.path.join(DATA_DIR, f"angles_chunk_{patient.name}_{i}.npy"), projection_angles_chunk)
            np.save(os.path.join(DATA_DIR, f"vector_field_chunk_{patient.name}_{i}.npy"), vector_field_chunk)




if __name__ == '__main__':
    sample_data()


