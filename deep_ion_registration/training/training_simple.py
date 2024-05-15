import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend('/home/j/J.Titze/pycharm_remote/remote_10/')


import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, LeaveOneOut
from tqdm import tqdm
from deep_ion_registration.helpers.plotting import plot_comparison
import numpy as np
from matplotlib import pyplot as plt
import pickle
from deep_ion_registration.networks.primal_dual_straight import PrimalDualStraight
from deep_ion_registration.dataset import IonDataset, load_system_matrices
from deep_ion_registration.dataset import PATIENTS
from pathlib import Path
import time
from deep_ion_registration.helpers.training_utils import TrainingLogger, Saver, EarlyStopping
from matplotlib import pyplot as plt



def loo_evaluation():
    base_dir = '/project/med6/IONCT/julian_titze'
    data_dir = base_dir + '/data/medical'
    path_system_matrices_normalized = data_dir + '/system_matrices_norm.pt'
    model_dir = base_dir + f'/models/{time.strftime("%Y-%m-%d_%H-%M-%S")}'

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    patient_names = np.array(PATIENTS)

    system_matrices_normalized = torch.load(path_system_matrices_normalized)

    learning_rate = 0.001
    num_epochs = 1000
    batch_size = 8
    patience = 5
    debug = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loo = LeaveOneOut()

    for fold, (train_ids, val_ids) in enumerate(loo.split(patient_names)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        model = PrimalDualStraight(system_matrix_normalized=system_matrices_normalized,
                                   n_iter=10,
                                   n_slices=5,
                                   n_angles=8, )

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model = model.to(device)
        logger = TrainingLogger(verbose=True)
        early_stopping = EarlyStopping(patience=patience)
        saver = Saver(model_dir)

        train_subsampler = IonDataset(data_dir, patient_names[train_ids], debug=debug)
        val_subsampler = IonDataset(data_dir, patient_names[val_ids])

        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

        train_model(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    logger=logger,
                    early_stopping=early_stopping,
                    saver=saver,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=num_epochs,
                    identifier=fold)

def train_model(model, criterion, optimizer, device, logger,
                early_stopping, saver, train_loader, val_loader, num_epochs, identifier):

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

            logger.log(epoch=epoch,
                       **{f'{phase}_loss': epoch_loss})

            if phase == 'val':
                stop_status = early_stopping(epoch_loss)
                if stop_status == 'stop':
                    saver.save(model, logger, identifier)
                    return model
                elif stop_status == 'divergence':
                    return model

        if epoch % 10 == 0:
            saver.save(model, logger, identifier)
            logger.log(epoch=epoch,
                       **{f'{phase}_examples': (outputs[0,2,:,:], labels[0,2,:,:])})

    saver.save(model, logger, identifier)

    return model


if __name__ == '__main__':
    loo_evaluation()