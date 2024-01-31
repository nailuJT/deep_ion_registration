import numpy as np
from datapipe.data_transform import GaussianParameters
from datapipe.data_transform import apply_gaussian_transform3d

class GaussianParameterSampler:
    def __init__(self, alpha_mean,
                 alpha_std,
                 mu_mean,
                 mu_std,
                 sigma_mean,
                 sigma_std,
                 rotation_mean,
                 rotation_std,
                 dimension=3):

        self.alpha_mean = np.array(alpha_mean)
        self.alpha_std = np.array(alpha_std)
        self.mu_mean = np.array(mu_mean)
        self.mu_std = np.array(mu_std)
        self.sigma_mean = np.array(sigma_mean)
        self.sigma_std = np.array(sigma_std)
        self.rotation_mean = np.array(rotation_mean)
        self.rotation_std = np.array(rotation_std)
        self.dimension = np.array(dimension)

        self.correlation_deformation = 0.1
        correlation_matrix_deformation = np.full((self.dimension, self.dimension), self.correlation_deformation)
        np.fill_diagonal(correlation_matrix_deformation, )
        self.correlation_matrix_deformation_full = np.kron(np.eye(self.dimension), correlation_matrix_deformation)

        self.correlation_directions = 0.8
        correlation_matrix_directions = np.eye(self.dimension) * self.correlation_directions

        self.correlation_matrix_directions_full = np.kron(np.ones(self.dimension) - np.eye(self.dimension), correlation_matrix_directions)

    def sample(self):
        alpha_directions = np.random.normal(self.alpha_mean.flatten(), self.alpha_std, self.dimension)

        correlation_coefficient = 0.8
        covariance_matrix = np.full((self.dimension, self.dimension), correlation_coefficient)
        np.fill_diagonal(covariance_matrix, 1)

        mu_directions = np.random.multivariate_normal(self.mu_mean, covariance_matrix,
                                                      (self.dimension, self.dimension))
        sigma_directions = np.random.multivariate_normal(self.sigma_mean, covariance_matrix,
                                                         (self.dimension, self.dimension))
        rotation_directions = np.random.multivariate_normal(self.rotation_mean, covariance_matrix,
                                                            self.dimension)

        return {"alpha_dirs": alpha_directions, "mu_dirs": mu_directions, "sigma_dirs": sigma_directions,
                "rotation_dirs": rotation_directions}
    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()

    @classmethod
    def from_config(cls, config):
        return cls(config['alpha_mean'],
                   config['alpha_std'],
                   config['mu_mean'],
                   config['mu_std'],
                   config['sigma_mean'],
                   config['sigma_std'],
                   config['rotation_mean'],
                   config['rotation_std'])


def transform_projection(projection, gaussian_parameters, normalize=True):
    """
    Samples a Gaussian transform and applies it to a projection.
    """
    ct_original = projection.patient.ct
    mask_original = projection.patient.mask
    voxel_size = projection.voxel_size

    if normalize:
        #TODO: implement normalization based on voxel size and image size
        pass

    ct_transformed, vector_field = apply_gaussian_transform3d(ct_original, **gaussian_parameters.__dict__)
    mask_transformed, _ = apply_gaussian_transform3d(mask_original, **gaussian_parameters.__dict__)
    projection.patient.ct = ct_transformed
    projection.patient.mask = mask_transformed

    return projection, vector_field



def test_sampler():
    sampler = GaussianParameterSampler(0.5, 0.5, 0, 1, 0, 1, 0, 1, 2)
    print(sampler.sample())


if __name__ == '__main__':
    test_sampler()


