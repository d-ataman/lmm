from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import onmt
from onmt.Utils import aeq
import torch.distributions as tdist


class Sampler(nn.Module):
    """
    The inference network based on MLP to learn the parameters of a diagonal
    Gaussian distribution and predict samples from it given an input.
    """

    def __init__(self, latent_dim, hidden_size):
        super(Sampler, self).__init__()

    def forward(self, X, batch_size, translate):
        out = self.run_forward_pass(X, batch_size, translate)
        return out



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True) 
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        return out

class MLP_SP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_SP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.softplus(out)
        return out


"""""""""""""""""""""""""""""""""""""""""""""""
Functions for the continuous Gaussian variable.

"""""""""""""""""""""""""""""""""""""""""""""""

class DiagonalGaussianSampler(Sampler):
    """
    The inference network based on MLP to learn the parameters of a diagonal 
    Gaussian distribution and predict samples from it given an input.
    """

    def __init__(self, latent_dim, hidden_size):
        super(DiagonalGaussianSampler, self).__init__(latent_dim, hidden_size)

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

        self.mu = MLP(self.hidden_size, self.hidden_size//2, self.latent_dim)
        self.sigma = MLP_SP(self.hidden_size, self.hidden_size//2, self.latent_dim)


    def sample_value(self, mean, variance, batch_size):
        """
        Produce a sample from the inferred Gaussian distribution.
        :param mean: The mean of the Gaussian.
        :param scale: The scale parameter of this Gaussian.
        :return: A random Gaussian vector.
        """

        N = tdist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        e = N.sample(sample_shape=torch.Size([batch_size, self.latent_dim]))
        return mean + variance * e.squeeze(2).cuda()

    def run_forward_pass(self, X, batch_size, translate):
        """
        Method for passing the input to the inference network
        """
        self.mean = self.mu(X)
        self.variance = self.sigma(X)
        if translate == False:
            s = self.sample_value(self.mean, self.variance, batch_size)
            return s
        else:
            return self.mean



"""""""""""""""""""""""""""""""""""""""""""""""
Functions for the discrete Kumaraswamy variables.

"""""""""""""""""""""""""""""""""""""""""""""""
def hardsigmoid(x):
    return torch.min(torch.ones_like(x), torch.max(x, torch.zeros_like(x)))

class RV:

    def params(self):
        raise NotImplementedError('Implement me')

    def sample(self, size=None):
        raise NotImplementedError('Implement me')
    
    def log_pdf(self, x):
        raise NotImplementedError('Implement me')
    
    def log_cdf(self, x):
        raise NotImplementedError('Implement me')
        
    def entropy(self):
        raise NotImplementedError('Implement me')
    
    def pdf(self, x):
        return torch.exp(self.log_pdf(x))
    
    def cdf(self, x):
        return torch.exp(self.log_cdf(x))


class RelaxedBinary(RV):
    
    pass


class Kuma(RelaxedBinary):

    def __init__(self, params: list):
        self.a = params[0]
        self.b = params[1]
        
    def params(self):
        return [self.a, self.b]

    def sample(self, size=None, eps=0.001):
        y = (2*eps - 1.) * torch.rand(size) + 1. - eps
        y = y.cuda()
        z = (1 - (1 - y).pow(1. / self.b)).pow(1. / self.a)
        return z    

    def log_pdf(self, x):
        t1 = torch.log(self.a) + torch.log(self.b) 
        t2 = (self.a - 1) * torch.log(x + 0.001)
        t3 = (self.b - 1) * torch.log(1. - torch.min(torch.pow(x, self.a), torch.tensor([0.999]).cuda()))
        return t1 + t2 + t3    
    
    def log_cdf(self, x):
        return torch.log(1. - torch.min(torch.pow((1. - torch.pow(x, self.a)), self.b), torch.tensor([0.999]).cuda()))

class StretchedVariable(RelaxedBinary):
    
    def __init__(self, dist: RelaxedBinary, support: list):
        """
        :param dist: a RelaxedBinary variable (e.g. BinaryConcrete or Kuma)
        :param support: a pair specifying the limits of the stretched support (e.g. [-1, 2])
            we use these values to compute location = pair[0] and scale = pair[1] - pair[0]        
        """
        assert isinstance(dist, RelaxedBinary), 'I need a RelaxedBinary variable, got %s' % type(dist)
        assert support[0] < support[1], 'I need an ordered support, got %s' % support
        self._dist = dist
        self.loc = support[0]
        self.scale = support[1] - support[0]
        
    def params(self):
        return self._dist.params()
        
    def sample(self, size=None):
        # sample a relaxed binary variable
        x_ = self._dist.sample(size=size)
        # and stretch it
        return x_ * self.scale + self.loc
    
    def log_pdf(self, x):
        # shrink the stretched variable
        x_ = (x - self.loc) / self.scale
        # and assess the stretched pdf using the original pdf 
        # see eq 25 (left) of Louizos et al
        return self._dist.log_pdf(x_) - torch.log(torch.tensor([self.scale]).cuda())
    
    def log_cdf(self, x):
        # shrink the stretched variable
        x_ = (x - self.loc) / self.scale
        # assess its cdf
        # see eq 25 (right) of Louizos et al
        return self._dist.log_cdf(x_)


class HardBinary(RV):
    
    def __init__(self, dist: StretchedVariable):
        assert isinstance(dist, StretchedVariable), 'I need a stretched variable'
        self._dist = dist
        
    def params(self):
        return self._dist.params()
        
    def sample(self, size=None):
        # sample a stretched variable
        x_ = self._dist.sample(size=size)        
        # and rectify it
        return hardsigmoid(x_)
    
    def log_pdf(self, x):
        # first we fix log_pdf for 0s and 1s
        log_p = torch.where(
            x == 0., 
            self._dist.log_cdf(0.),  # log Q(0) 
            torch.log(1. - self._dist.cdf(1.))  # log (1-Q(1))
        )

        # then for those that are in the open (0, 1)
        log_p = torch.where(
            torch.lt(x, 0.) * torch.lt(x, 1.),
            torch.log(self._dist.cdf(1.) - self._dist.cdf(0.)) + self._dist.log_pdf(x),
            log_p
        )
        # see eq 26 of Louizos et al
        return log_p
    
    def log_cdf(self, x):        
        log_c = torch.where(
            torch.lt(x, 1.), 
            self._dist.log_cdf(x),
            torch.full_like(x, 0.)  # all of the mass
        )
        return log_c


class HardKuma(HardBinary):
    
    def __init__(self, params: list, support: list):
        super(HardKuma, self).__init__(StretchedVariable(Kuma(params), support))


class KumaSampler(Sampler):
    """
    The inference network based on MLP to learn the parameters of a discrete
    Kumaraswamy distribution and predict samples from it given an input.
    """

    def __init__(self, latent_dim, hidden_size):
        super(KumaSampler, self).__init__(latent_dim, hidden_size)

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

        self.na = MLP_SP(self.hidden_size, self.hidden_size//2, self.latent_dim)
        self.nb = MLP_SP(self.hidden_size, self.hidden_size//2, self.latent_dim)


    def sample(self, a, b, size):
        """
        Produce a sample from the Kumaraswamy distribution.
        """
        k = HardKuma([a, b], [-0.1, 1.1]) # support of the stretched variable should be just a bit bigger than the base Kumaraswamy
        ksample = k.sample(size=size)
        logpdfloss = torch.log(sum(sum(1 - k.pdf(torch.Tensor([1.]).cuda()) - k.pdf(torch.Tensor([0.]).cuda())))/float(ksample.size(0)*self.latent_dim))
        return ksample, logpdfloss

    def run_forward_pass(self, X, batch_size, translate):
        """
        Method for passing the input to the inference network
        """
        #out_s = []; out_logpdfloss = []
        self.a = self.na(X)
        self.b = self.nb(X)
        s, logpdfloss = self.sample(self.a, self.b, size=1)
        if translate == False:
            return s, logpdfloss
        else:
            return s, logpdfloss
