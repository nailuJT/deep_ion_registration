import numpy as np
import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
from deep_ion_registration.dataset import IonDataset, load_system_matrices

def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


class ConcatenateLayer(nn.Module):
    def __init__(self):
        super(ConcatenateLayer, self).__init__()

    @staticmethod
    def forward(*x):
        return torch.cat(list(x), dim=1)

class DualNet(nn.Module):
    def __init__(self, n_ions):
        super(DualNet, self).__init__()

        self.n_channels = 3 * n_ions
        self.input_concat_layer = ConcatenateLayer()
        layers = [
            nn.Conv2d(self.n_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, n_ions, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, h, Op_f, g):
        x = self.input_concat_layer(h, Op_f, g)
        x = h + self.block(x)
        return x

class PrimalNet(nn.Module):
    def __init__(self, n_primal):
        super(PrimalNet, self).__init__()

        self.n_primal = n_primal
        self.n_channels = n_primal + 1
        self.input_concat_layer = ConcatenateLayer()
        layers = [
            nn.Conv2d(self.n_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, self.n_primal, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, f, OpAdj_h):
        x = self.input_concat_layer(f, OpAdj_h)
        x = f + self.block(x)
        return x


class PrimalDualStraight(nn.Module):
    def __init__(self,
                 system_matrix=None,
                 system_matrix_normalized=None,
                 primal_architecture=PrimalNet,
                 dual_architecture=DualNet,
                 n_iter=10,
                 n_primal=5,
                 n_ions=1,
                 checkpointing=False):

        super(PrimalDualStraight, self).__init__()
        self.primal_architecture = primal_architecture
        self.dual_architecture = dual_architecture
        self.n_iter = n_iter
        self.n_primal = n_primal
        self.checkpointing = checkpointing

        self.primal_nets = nn.ModuleList()
        self.dual_nets = nn.ModuleList()

        for i in range(n_iter):
            self.primal_nets.append(
                primal_architecture(n_primal)
            )
            self.dual_nets.append(
                dual_architecture(n_ions)
            )

        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        self.system_matrix = system_matrix
        self.system_matrix_normalized = system_matrix_normalized

    def forward_rework(self, dual, primal, projection, device, logger):
        system_tensors_norm = self.system_matrix_normalized.to_dense().to(device)

        for k in np.arange(self.system_matrix):
            primal_new = reshape_fortran(primal[:, :, :, :, :], (
                primal.shape[0], primal.shape[1], primal.shape[-3] * primal.shape[-2] * primal.shape[-1]))

            evalop1 = torch.einsum('ijkl, mnk -> mnil', system_tensors_norm, primal_new)

            dual= self.dual_forward(dual, evalop1, projection, k, self.dummy_tensor)

            evalop2 = torch.einsum('ijkl, mnil -> mnik', dual, system_tensors_norm)
            evalop2 = torch.sum(evalop2, dim=-2)
            evalop2 = reshape_fortran(evalop2, evalop2.shape[0], evalop2.shape[1], primal.shape[-3], primal.shape[-2])

            primal = self.primal_forward(primal.squeeze(dim=4), evalop2.squeeze(dim=4), k, self.dummy_tensor)

        return primal


    def forward(self, dual, primal, projection, device, return_all):
        # primal.requires_grad=True ##warum
        if return_all:
            forward_all = torch.empty([self.n_iter] + list(dual.shape))
            backward_all = torch.empty([self.n_iter] + list(primal.shape))

        system_tensors = self.system_matrix
        system_tensors_norm = self.system_matrix_normalized  # normalisation of system matrix (to avoid vanishing (both forward and back normalised) or explosion (both not normalised) of image values) in forward projection

        for k in np.arange(self.n_iter):

            evalop1 = torch.empty(projection.shape).to(device)
            primal_new = primal.reshape(primal.shape[0], primal.shape[1], primal.shape[2] * primal.shape[3])

            primal_new_new = reshape_fortran(primal[:, :, :, :, :], (
                primal.shape[0], primal.shape[1], primal.shape[-3] * primal.shape[-2] * primal.shape[-1]))

            evalop_rework = torch.einsum('ijkl, mnk -> mnil', system_tensors_norm.to_dense(), primal_new)

            for j in np.arange(dual.shape[2]):
                sys_tensor = system_tensors_norm[j]
                # print(sys_tensor.shape)
                # print(primal.shape, 'primal')
                # sys_batches = system_tensors_norm.transpose(0,1)[j].to(device)
                sys_batches = sys_tensor.permute(0, 2, 1).to(device)
                img_batches = reshape_fortran(primal[:, 1, :, :, :], (
                    primal.shape[0], primal.shape[-3] * primal.shape[-2] * primal.shape[-1], 1))
                prod = torch.bmm(sys_batches, img_batches)[:, :, 0]
                prod = prod.reshape(prod.shape[0], primal.shape[-3],
                                    int(prod.shape[1] / primal.shape[-3])).transpose(2, 1)
                evalop1[:, :, j, :] = prod

            if self.checkpointing:
                dual = checkpoint(self.dual_forward, dual, evalop1, projection, k, self.dummy_tensor)
            else:
                dual = self.dual_forward(dual, evalop1, projection, k, self.dummy_tensor)
            if return_all:
                forward_all[k, :, :, :, :] = dual

            evalop2 = torch.empty(
                (dual.shape[2], primal.shape[0], 1, primal.shape[2], primal.shape[3], primal.shape[4])).to(device)
            for j in np.arange(dual.shape[2]):
                sys_tensor = system_tensors[j]
                sys_batch = sys_tensor.to(device)
                # sys_batch = system_tensors.transpose(0,1)[j].to(device)
                dual2 = dual.transpose(1, 2).transpose(2, 3)
                dual2 = dual2.reshape(dual2.shape[0], dual2.shape[1], dual2.shape[2] * dual2.shape[3])
                prod = reshape_fortran(torch.bmm(sys_batch, dual2[:, j, :, None]),
                                       (primal.shape[0], primal.shape[2], primal.shape[3], primal.shape[4]))

                evalop2[j, :, 0, :, :, :] = prod
            evalop2 = torch.sum(evalop2, dim=0)
            if self.checkpointing:
                primal = checkpoint(self.primal_forward, primal.squeeze(dim=4), evaqlop2.squeeze(dim=4), k,
                                    self.dummy_tensor)
            else:
                primal = self.primal_forward(primal.squeeze(dim=4), evalop2.squeeze(dim=4), k, self.dummy_tensor)
            primal = primal.unsqueeze(dim=4)
            if return_all:
                backward_all[k, :, :, :, :] = primal
        if return_all:
            return primal[:, 0:1, :, :], forward_all, backward_all
        else:
            return primal[:, 0:1, :, :]

    def primal_forward(self, primal, evalop2, k, dummy_tensor):
        return self.primal_nets[k](primal, evalop2)

    def dual_forward(self, dual, evalop1, g, k, dummy_tensor):
        return self.dual_nets[k](dual, evalop1, g)


if __name__ == '__main__':
    from pathlib import Path
    data_dir = '/project/med6/IONCT/julian_titze/data/raw'

    path_system_matrices = Path.joinpath(Path(data_dir), 'system_matrices.pt')
    system_matrices = torch.load(path_system_matrices)
    system_matrices = system_matrices.unsqueeze(1)
    path_system_matrices_normalized = Path.joinpath(Path(data_dir), 'system_matrices_norm.pt')
    system_matrices_normalized = torch.load(path_system_matrices_normalized)
    system_matrices_normalized = system_matrices_normalized.unsqueeze(1)

    dataset = IonDataset(data_dir)
    xray_ct, mask, projection_angle, transformed_ion_ct = dataset[0]

    norm_val = 1

    primal = torch.as_tensor(xray_ct.copy() / norm_val, dtype=torch.float)
    #add channel dimension at the end and batch dimension
    primal = primal.unsqueeze(0).unsqueeze(-1)

    g = projection_angle[:,0,:]
    #add batch dimension and channel dimension
    g = g.unsqueeze(0).unsqueeze(0)

    h0 = np.zeros(g.shape)
    dual = torch.as_tensor(h0.copy(), dtype=torch.float)

    model = PrimalDualStraight(system_matrix=system_matrices,
                               system_matrix_normalized=system_matrices_normalized,
                               n_iter=10,
                               n_primal=11,
                               n_ions=1,
                               checkpointing=False)

    y = model(dual, primal, g, 'cpu', False)
    print(y.shape)

    print(model)