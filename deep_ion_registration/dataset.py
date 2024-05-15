import torch
from torch.utils.data import Dataset
import numpy as np
import os
from deep_ion_registration.helpers.plotting import compare_images

PATIENTS = ['male1', 'female1', 'male2','female2','male3', 'female3', 'male4', 'female4', 'male5', 'female5']

class IonDataset(Dataset):
    def __init__(self, data_dir, human_names=None, debug=False):
        self.data_dir = data_dir
        self.human_names = human_names

        self.ionct_files = self.prepare_data('ionct_chunk')
        self.xray_ct = self.prepare_data('ct_chunk')
        self.vector_field_files = self.prepare_data('vector_field_chunk')
        self.mask_files = self.prepare_data('mask_chunk')
        self.projection_angle_files = self.prepare_data('angles_chunk')
        self.transformed_ionct_files = self.prepare_data('transformed_ionct_chunk')
        self.debug = debug

    def __len__(self):
        return len(self.ionct_files)

    def prepare_data(self, name):
        data_index = sorted([f for f in os.listdir(self.data_dir) if f.startswith(name)])

        if self.human_names is not None:
            data_index = [f for f in data_index if any(human_name in f for human_name in self.human_names)]

        return data_index

    def __getitem__(self, idx):
        xray_ct_path = os.path.join(self.data_dir, self.xray_ct[idx])
        projection_angle_path = os.path.join(self.data_dir, self.projection_angle_files[idx])
        transformed_ionct_path = os.path.join(self.data_dir, self.transformed_ionct_files[idx])

        xray_ct = np.load(xray_ct_path)
        projection_angle = np.load(projection_angle_path)
        transformed_ionct = np.load(transformed_ionct_path)

        # convert to torch tensor
        xray_ct = torch.from_numpy(xray_ct).float()
        projection_angle = torch.from_numpy(projection_angle).float()
        dual_initial_guess = torch.zeros_like(projection_angle)
        transformed_ionct = torch.from_numpy(transformed_ionct).float()

        inputs = (xray_ct, projection_angle, dual_initial_guess)

        if self.debug:
            compare_images(xray_ct[0], transformed_ionct[0])

        return inputs, transformed_ionct

def load_system_matrices(path):
    system_matrices = torch.load(path)
    return system_matrices

def test_custom_dataset():
    data_dir = '/project/med6/IONCT/julian_titze/data/medical'
    dataset = IonDataset(data_dir, human_names=PATIENTS)

    shape_xray_ct = (5, 256, 256)
    shape_projection = (8, 5, 256)
    shape_dual_initial_guess = (8, 5, 256)

    for index, data in enumerate(dataset):
        print(index)
        (xray_ct, projection_angle, dual_initial_guess), transformed_ionct = data
        # print(xray_ct.shape)
        assert xray_ct.shape == shape_xray_ct
        assert transformed_ionct.shape == shape_xray_ct
        assert projection_angle.shape == shape_projection
        assert dual_initial_guess.shape == shape_dual_initial_guess


if __name__ == '__main__':
    test_custom_dataset()
