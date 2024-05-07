import torch
from deep_ion_registration.networks.primal_dual_straight import PrimalDualStraight, DualNet, PrimalNet
from deep_ion_registration.dataset import IonDataset


def test_dual_net():
    n_slices = 11
    n_angles = 5
    dual_net = DualNet(n_slices)
    dual = torch.ones(1, n_slices, n_angles, 128)
    primal_projected = torch.ones(1, n_slices, n_angles, 128)
    projections = torch.ones(1, n_slices, n_angles, 128)
    x = dual_net(dual, primal_projected, projections)
    print(x.shape)


def test_primal_net():
    n_slices = 11
    primal_net = PrimalNet(n_slices)
    primal = torch.ones(1, n_slices, 128, 128)
    dual_projected = torch.ones(1, n_slices, 128, 128)
    reference_ct = torch.ones(1, n_slices, 128, 128)
    x = primal_net(primal, dual_projected, reference_ct)
    print(x.shape)


def test_primal_dual_straight():
    test_dual_net()
    test_primal_net()

    from pathlib import Path
    data_dir = '/project/med6/IONCT/julian_titze/data/raw'

    path_system_matrices = Path.joinpath(Path(data_dir), 'system_matrices.pt')
    system_matrices = torch.load(path_system_matrices)

    path_system_matrices_normalized = Path.joinpath(Path(data_dir), 'system_matrices_norm.pt')
    system_matrices_normalized = torch.load(path_system_matrices_normalized)

    dataset = IonDataset(data_dir)
    (reference_ct, projection, dual_initial_guess), transformed_ion_ct = dataset[0]

    # include batch dimension
    reference_ct = reference_ct.unsqueeze(0)
    projection = projection.unsqueeze(0)
    dual_initial_guess = dual_initial_guess.unsqueeze(0)

    primal_dual_straight = PrimalDualStraight(system_matrix=system_matrices,
                                              system_matrix_normalized=system_matrices_normalized,
                                              n_iter=10,
                                              n_slices=11)
    ion_ct = primal_dual_straight(reference_ct, projection, dual_initial_guess, logger=None)
    print(ion_ct.shape)


if __name__ == '__main__':
    test_primal_dual_straight()