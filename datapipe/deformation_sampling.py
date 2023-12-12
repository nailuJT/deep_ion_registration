import numpy as np
from data_transform import GaussianParameters
from data_transform import gaussian_shift

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

        self.alpha_mean = alpha_mean
        self.alpha_std = alpha_std
        self.mu_mean = mu_mean
        self.mu_std = mu_std
        self.sigma_mean = sigma_mean
        self.sigma_std = sigma_std
        self.rotation_mean = rotation_mean
        self.rotation_std = rotation_std
        self.dimension = dimension

    def sample(self):
        alpha_directions = np.random.normal(self.alpha_mean, self.alpha_std, self.dimension)
        mu_directions = np.random.normal(self.mu_mean, self.mu_std, (self.dimension, self.dimension))
        sigma_directions = np.random.normal(self.sigma_mean, self.sigma_std, (self.dimension, self.dimension))
        rotation_directions = np.random.normal(self.rotation_mean, self.rotation_std, self.dimension)

        return GaussianParameters(alpha_directions, mu_directions, sigma_directions, rotation_directions)

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


def test_sampler():
    sampler = GaussianParameterSampler(0.5, 0.5, 0, 1, 0, 1, 0, 1, 2)
    print(sampler.sample())


if __name__ == '__main__':
    test_sampler()


