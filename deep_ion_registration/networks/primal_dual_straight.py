import numpy as np
import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
from deep_ion_registration.dataset import IonDataset, load_system_matrices


class DualNet(nn.Module):
    def __init__(self, n_angles):
        super(DualNet, self).__init__()

        self.n_channels = 3 * n_angles

        layers = [
            nn.Conv2d(self.n_channels, 32, kernel_size=5, padding='same'),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding='same'),
            nn.PReLU(),
            nn.Conv2d(32, n_angles, kernel_size=5, padding='same'),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, dual, primal_projected, projections):
        x = torch.cat((dual, primal_projected, projections), dim=1)
        x = dual + self.block(x)
        return x


class PrimalNet(nn.Module):
    def __init__(self, n_slices):
        super(PrimalNet, self).__init__()

        self.n_primal = n_slices
        self.n_channels = 2 * n_slices

        layers = [
            nn.Conv2d(self.n_channels, 32, kernel_size=5, padding='same'),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding='same'),
            nn.PReLU(),
            nn.Conv2d(32, self.n_primal, kernel_size=5, padding='same')
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, primal, dual_projected):
        x = torch.cat((primal, dual_projected), dim=1)
        x = primal + self.block(x)
        return x


class PrimalDualStraight(nn.Module):
    """
    Primal Dual network for straight projection implementation.
    """
    def __init__(self,
                 system_matrix_normalized,
                 n_slices,
                 n_angles,
                 image_shape=None,
                 n_iter=10,
                 primal_architecture=PrimalNet,
                 dual_architecture=DualNet):
        """
        :param system_matrix: system matrix of projection
        :param system_matrix_normalized: normalized along projection axis
        :param primal_architecture:
        :param dual_architecture:
        :param n_iter: iterations of the primal dual algorithm
        :param n_slices: slices of the CT block
        :param n_angles: projection angles
        :param image_shape: shape of the ct slice
        """

        super(PrimalDualStraight, self).__init__()
        self.primal_architecture = primal_architecture
        self.dual_architecture = dual_architecture
        self.n_iter = n_iter
        self.n_slices = n_slices
        self.image_shape = image_shape
        self.primal_nets = nn.ModuleList()
        self.dual_nets = nn.ModuleList()

        for i in range(n_iter):
            self.primal_nets.append(
                primal_architecture(n_slices)
            )
            self.dual_nets.append(
                dual_architecture(n_angles)
            )

        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        self.system_matrix = system_matrix
        self.system_matrix_normalized = system_matrix_normalized

    def forward(self, refernce_ct, projection, dual_initial_guess,  device='cpu', logger=None):
        """
        :param refernce_ct: treatment planning CT
        :param projection: ion projection
        :param dual_initial_guess: initial guess for dual variable
        :param device: cuda or cpu
        :param logger: logger object
        :return: ion_ct
        """
        device = refernce_ct.device
        dual = dual_initial_guess
        system_tensors_norm = self.system_matrix_normalized.to_dense().to(device)
        primal = refernce_ct
        shape_primal_projected = (primal.shape[0], self.n_slices, system_tensors_norm.shape[1])
        shape_dual_projected = (dual.shape[0], self.n_slices, primal.shape[-2], primal.shape[-1])

        for k in np.arange(self.n_iter):
            # project primal into dual space
            primal_projected = primal.view(shape_primal_projected)
            primal_projected = torch.einsum('ikl, mnk -> minl', system_tensors_norm, primal_projected)

            dual = self.dual_forward(k, dual, primal_projected, projection)
            if logger:
                logger.log_dual(dual)

            # project dual back into primal space
            dual_back_projected = torch.einsum('ikl, minl -> mink', system_tensors_norm, dual)
            dual_back_projected = torch.sum(dual_back_projected, dim=1)
            dual_back_projected = dual_back_projected.view(shape_dual_projected)

            primal = self.primal_forward(k, primal, dual_back_projected, refernce_ct)
            if logger:
                logger.log_primal(primal)

        return primal

    def primal_forward(self, k, *args):
        return self.primal_nets[k](*args)

    def dual_forward(self, k, *args):
        return self.dual_nets[k](*args)


class LoggerPrimalDual:
    def __init__(self):
        self.primal_outputs = []
        self.dual_outputs = []

    def log_primal(self, output):
        self.primal_outputs.append(output.detach().cpu().numpy())

    def log_dual(self, output):
        self.dual_outputs.append(output.detach().cpu().numpy())

    def reset(self):
        self.primal_outputs = []
        self.dual_outputs = []