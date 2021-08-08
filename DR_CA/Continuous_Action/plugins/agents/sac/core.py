import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from ipdb import set_trace
import math

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

LOG_STD_MAX = 2

LOG_STD_MIN = -20

class SquashedGaussianMLPActor_rlpyt(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

        self.EPS = 1e-8
        self.act_dim = act_dim
        self.squash = 1

    def log_likelihood(self, x, mean, std, log_std):

        z = (x - mean) / (std + self.EPS)
        logli = -(torch.sum(log_std + 0.5 * z ** 2, dim=-1) +
            0.5 * self.act_dim * math.log(2 * math.pi))
        logli -= torch.sum(torch.log(self.squash * (1 - torch.tanh(x) ** 2) + self.EPS),
            dim=-1)
        return logli

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # # Pre-squash distribution and sample
        if deterministic:
            # pi_distribution = Normal(mu, std)
            # Only used for evaluating policy at test time.
            pi_action = mu
            pi_action = torch.tanh(pi_action)
            return pi_action, None, torch.tensor([0]), torch.tanh(mu), std

        epsilon = None
        logp_pi = None
        if with_logprob:
            # ---- Spinnup
            # pi_distribution = Normal(mu, std)
            # pi_action = pi_distribution.rsample()
            # logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)

            # ---- rlpyt
            epsilon = torch.normal(torch.zeros_like(mu), torch.ones_like(mu))
            pi_action = mu + epsilon * std
            logp_pi = self.log_likelihood(pi_action, mu, std, log_std)

        else:
            # Another reparameterization trick borrowed from rlpyt implementation of SAC
            epsilon = torch.normal(torch.zeros_like(mu), torch.ones_like(mu))
            pi_action = mu + epsilon * std
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        return pi_action, logp_pi, epsilon, torch.tanh(mu), std

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # # Pre-squash distribution and sample
        if deterministic:
            # pi_distribution = Normal(mu, std)
            # Only used for evaluating policy at test time.
            pi_action = mu
            pi_action = torch.tanh(pi_action)
            return pi_action, None, torch.tensor([0]), torch.tanh(mu), std

        epsilon = None
        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            pi_distribution = Normal(mu, std)
            pi_action = pi_distribution.rsample()
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            # Another reparameterization trick borrowed from rlpyt implementation of SAC
            epsilon = torch.normal(torch.zeros_like(mu), torch.ones_like(mu))
            pi_action = mu + epsilon * std
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi, epsilon, torch.tanh(mu), std

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    # def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
    #              activation=nn.ReLU):
    #     super().__init__()

    def __init__(self, obs_ac_dim, obs_cr_dim, act_dim, hidden_sizes=(256, 256),
                     activation=nn.ReLU):
        super().__init__()

        #obs_ac_dim = obs_ac_dim
        #obs_cr_dim = obs_cr_dim

        act_dim = act_dim
        act_limit = act_dim

        # build policy and value functions
        # self.pi = SquashedGaussianMLPActor_rlpyt(obs_ac_dim, act_dim, hidden_sizes, activation, act_limit)
        self.pi = SquashedGaussianMLPActor(obs_ac_dim, act_dim, hidden_sizes, activation, act_limit)

        self.q1 = MLPQFunction(obs_cr_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_cr_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _, epsilon, mu, std = self.pi(obs, deterministic, False)
            return a.numpy(), epsilon.numpy(), mu.numpy(), std.numpy()
