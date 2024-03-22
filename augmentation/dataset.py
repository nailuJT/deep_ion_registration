import torch
from torch.utils.data import Dataset
import numpy as np
import os

from helpers.plotting import compare_images

class IonDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ionct_files = sorted([f for f in os.listdir(data_dir) if 'ionct_chunk' in f])
        self.label_files = sorted([f for f in os.listdir(data_dir) if 'vector_field_chunk' in f])
        self.mask_files = sorted([f for f in os.listdir(data_dir) if 'mask_chunk' in f])
        self.angle_files = sorted([f for f in os.listdir(data_dir) if 'angles_chunk' in f])
        self.transformed_ionct_files =

    def __len__(self):
        return len(self.ionct_files)

    def __getitem__(self, idx):
        ionct_path = os.path.join(self.data_dir, self.ionct_files[idx])
        label_path = os.path.join(self.data_dir, self.label_files[idx])
        mask_path = os.path.join(self.data_dir, self.mask_files[idx])
        angle_path = os.path.join(self.data_dir, self.angle_files[idx])

        ionct = np.load(ionct_path)
        label = np.load(label_path)
        mask = np.load(mask_path)
        angle = np.load(angle_path)

        # convert to torch tensor
        ionct = torch.from_numpy(ionct).float()
        label = torch.from_numpy(label).float()
        mask = torch.from_numpy(mask).float()
        angle = torch.from_numpy(angle).float()

        return ionct, label, mask, angle

def test_custom_dataset():
    data_dir = '/project/med6/IONCT/julian_titze/data/raw'
    dataset = IonDataset(data_dir)
    ionct, label, mask, angle = dataset[0]




if __name__ == '__main__':
    test_custom_dataset()
