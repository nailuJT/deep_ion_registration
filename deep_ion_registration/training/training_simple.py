import os
import sys

# Set working directory and update sys.path
os.chdir("/home/j/J.Titze/Projects/deep_ion_registration")
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, LeaveOneOut
from tqdm import tqdm
from ..helpers.plotting import plot_comparison
import numpy as np
from matplotlib import pyplot as plt


def train_model(model,
                dataset_class,
                data_dir,
                patient_names,
                criterion,
                optimizer,
                num_epochs=25,
                device=torch.device("cpu"),
                batch_size=32):

    model = model.to(device)

    loo = LeaveOneOut()

    for fold, (train_ids, val_ids) in enumerate(loo.split(patient_names)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_subsampler = dataset_class(data_dir, patient_names[train_ids])
        val_subsampler = dataset_class(data_dir, patient_names[val_ids])

        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                    dataloader = train_loader
                else:
                    model.eval()
                    dataloader = val_loader

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm(dataloader):
                    inputs = [input_item.to(device) for input_item in inputs]
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(*inputs)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs[0].size(0)

                epoch_loss = running_loss / len(dataloader.dataset)

                print(f'{phase} Loss: {epoch_loss:.4f}')

        #plot example
        plot_comparison(labels[0,2,:,:], outputs[0,2,:,:])

    return model

def main():
    from ..networks.primal_dual_straight import PrimalDualStraight
    from ..dataset import IonDataset, load_system_matrices
    from ..dataset import PATIENTS
    from pathlib import Path


    data_dir = '/project/med6/IONCT/julian_titze/data/medical'
    model_dir = '/project/med6/IONCT/julian_titze/models'
    patient_names = np.array(PATIENTS)

    path_system_matrices = Path.joinpath(Path(data_dir), 'system_matrices.pt')
    system_matrices = torch.load(path_system_matrices)

    path_system_matrices_normalized = Path.joinpath(Path(data_dir), 'system_matrices_norm.pt')
    system_matrices_normalized = torch.load(path_system_matrices_normalized)

    learning_rate = 0.001
    num_epochs = 100
    batch_size = 32
    n_splits = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")

    model = PrimalDualStraight(system_matrix=system_matrices,
                               system_matrix_normalized=system_matrices_normalized,
                               n_iter=10,
                               n_slices=5,
                               n_angles=8,)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trained_model = train_model(model=model,
                                dataset_class=IonDataset,
                                data_dir=data_dir,
                                patient_names=patient_names,
                                criterion=criterion,
                                optimizer=optimizer,
                                num_epochs=num_epochs,
                                device=device,
                                batch_size=batch_size)

    return trained_model

def save_trained_model(trained_model, path):
    torch.save(trained_model.state_dict(), path)

if __name__ == '__main__':
    main()