from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq
import torch.distributions as tdist

class ReparametrisationSampler(nn.Module):
    """
    Reparametrization Sampler for the Variational Autoencoder.

    Args:
    num_latent_vars: The number of latent variables
    latent_dim: dimension of the latent variable vector
    hidden_sze: number of hidden units at each layer
    num_affixes: number of hidden layers in the inference network 
    """

    def __init__(self, num_vars=1, latent_dim, hidden_size, num_affixes):
        self.num_vars = num_vars
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_affixes = num_affixes
        super(ReparametrisationSampler, self).__init__()


    def get_reparametrisation_sampler(self, num_vars=1, latent_dim, hidden_size, num_affixes):
        """
        Method for calling the sampler.
        Right now only the Diagonal Gaussian distribution is supported.
        """
        return DiagonalGaussianSampler(num_vars, latent_dim, hidden_size, num_affixes)


class DiagonalGaussianSampler(ReparametrisationSampler):
    """
    The inference network based on MLP to learn the parameters of a diagonal 
    Gaussian distribution and predict samples from it given an input.
    """

    def __init__(self, latent_dim, hidden_size):
        super(DiagonalGaussianSampler, self).__init__()

    self.latent_dim = latent_dim
    self.hidden_size = hidden_size


    self.mu = nn.ModuleList(nn.Linear(self.hidden_size, self.hidden_size, bias=True),
                                           torch.nn.ReLU(),
                                           nn.Linear(self.hidden_size, self.latent_dim, bias=True))

    self.sigma = nn.ModuleList(nn.Linear(self.hidden_size, self.hidden_size, bias=True),
                                           torch.nn.ReLU(),
                                           nn.Linear(self.hidden_size, self.latent_dim, bias=True))

    def sample_value(self, mean, variance):
        """
        Produce a sample from the inferred Gaussian distribution.
        :param mean: The mean of the Gaussian.
        :param scale: The scale parameter of this Gaussian.
        :return: A random Gaussian vector.
        """

        N = tdist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        e = N.sample(sample_shape=self.latent_dim)
        return mean + variance * e

    def forward(self, X):
    """
    Method for passing the input to the inference network
    """
        mean = self.mu(X)
        variance = self.sigma(X)
        z = sample_value(self, mean, variance)
        return z


