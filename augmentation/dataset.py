import torch
from torch.utils.data import Dataset
import numpy as np
import os

from helpers.plotting import compare_images

class IonDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ionct_files = sorted([f for f in os.listdir(data_dir) if 'ionct_chunk' in f])
        self.xray_ct = sorted([f for f in os.listdir(data_dir) if 'ct_chunk' in f])
        self.vector_field_files = sorted([f for f in os.listdir(data_dir) if 'vector_field_chunk' in f])
        self.mask_files = sorted([f for f in os.listdir(data_dir) if 'mask_chunk' in f])
        self.projection_angle_files = sorted([f for f in os.listdir(data_dir) if 'angles_chunk' in f])
        self.transformed_ionct_files = sorted([f for f in os.listdir(data_dir) if 'transformed_ionct_chunk' in f])

    def __len__(self):
        return len(self.ionct_files)

    def __getitem__(self, idx):
        ionct_path = os.path.join(self.data_dir, self.ionct_files[idx])
        xray_ct_path = os.path.join(self.data_dir, self.xray_ct[idx])
        vector_path = os.path.join(self.data_dir, self.vector_field_files[idx])
        mask_path = os.path.join(self.data_dir, self.mask_files[idx])
        projection_angle_path = os.path.join(self.data_dir, self.projection_angle_files[idx])
        transformed_ionct_path = os.path.join(self.data_dir, self.transformed_ionct_files[idx])

        ionct = np.load(ionct_path)
        xray_ct = np.load(xray_ct_path)
        vector_field = np.load(vector_path)
        mask = np.load(mask_path)
        projection_angle = np.load(projection_angle_path)
        transformed_ionct = np.load(transformed_ionct_path)

        # convert to torch tensor
        ionct = torch.from_numpy(ionct).float()
        vector_field = torch.from_numpy(vector_field).float()
        mask = torch.from_numpy(mask).float()
        projection_angle = torch.from_numpy(projection_angle).float()
        transformed_ionct = torch.from_numpy(transformed_ionct).float()

        return xray_ct, mask, projection_angle, transformed_ionct

def test_custom_dataset():
    data_dir = '/project/med6/IONCT/julian_titze/data/raw'
    dataset = IonDataset(data_dir)
    ionct, label, mask, angle = dataset[0]
    print(ionct.shape, label.shape, mask.shape, angle.shape)




if __name__ == '__main__':
    test_custom_dataset()
