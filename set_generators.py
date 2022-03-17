import torch
from torch import Tensor
import torch.nn as nn
from layers.base_layers import MLP
from utils import bridson
import numpy as np
import math
average_n = 11.0


def round_n(predicted_n, max_n):
    if predicted_n.dtype != torch.int64:
        predicted_n = torch.round(predicted_n)
    n = int(predicted_n.item())
    if n <= 1:
        n = 2
    if n >= max_n:
        n = max_n - 1
    return n


class SetGenerator(nn.Module):
    def __init__(self, cfg):
        """ Base class for a set generator. During training, the number of points n is assumed to be given.
            At generation time, if learn_from_latent, a value is predicted.
            Otherwise, a value is sampled from the train distribution
            n_distribution: dict. Each key is an integer, and the value the frequency at which sets of this
            size appear in the training set. """
        super().__init__()
        self.latent_channels = cfg.latent_dim
        self.set_channels = cfg.set_channels

        self.learn_from_latent = cfg.learn_from_latent
        self.predict_molecular_formula = cfg.predict_molecular_formula

        self.n_distribution = cfg.n_distribution
        self.n_probs = cfg.n_prob
        self.max_n = cfg.max_n
        self.extrapolation_n = cfg.extrapolation_n
        self.dataset_max_n = cfg.dataset_max_n
        self.dummy_param = nn.Parameter(torch.empty(0))     # Used to store the device
        if self.learn_from_latent:
            self.mlp1 = MLP(self.latent_channels, 1, cfg.hidden, cfg.num_mlp_layers)
        if self.predict_molecular_formula:
            self.mlp2 = MLP(self.latent_channels, cfg.num_atom_types, cfg.hidden, cfg.num_mlp_layers)

    def forward(self, latent: Tensor, n: int = None, extrapolation=False):
        """ A set generator returns a latent set with n nodes and set_channels features.
        Input: latent (Tensor of size batch x latent_channels)
        Returns: x (Tensor of size batch x n x set_channels).
                 n: int
        """
        predicted_n = self.mlp1(latent).squeeze(1) + average_n if (self.learn_from_latent and n is not None) else None
        if n is None:
            n = self.generate_n(latent, extrapolation)
        predicted_formula = torch.softmax(self.mlp2(latent), dim=1) if self.predict_molecular_formula else None

        return n, predicted_n, predicted_formula

    def generate_n(self, z: Tensor = None, extrapolation = False):
        n = self.mlp1(z) + average_n if self.learn_from_latent else torch.multinomial(self.n_probs, num_samples=1)
        if extrapolation:
            n = n + self.extrapolation_n
        return round_n(n, self.dataset_max_n)


class MLPGenerator(SetGenerator):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.mlp_gen_hidden = cfg.mlp_gen_hidden
        self.mlp = MLP(self.latent_channels, self.max_n * self.set_channels, self.mlp_gen_hidden, nb_layers=2)

    def forward(self, latent: Tensor, n: int = None, extrapolation=False):
        batch_size = latent.shape[0]
        n, predicted_n, predicted_formula = super().forward(latent, n, extrapolation)

        points = self.mlp(latent).reshape(batch_size, self.max_n, self.set_channels)
        points = points[:, :n, :]
        return points, predicted_n, predicted_formula


class RandomSetGenerator(SetGenerator):
    def forward(self, latent: Tensor, n: int = None, extrapolation=False):
        batch_size = latent.shape[0]
        n, predicted_n, predicted_formula = super().forward(latent, n, extrapolation)

        points = torch.randn(batch_size, n, self.set_channels, dtype=torch.float).to(self.dummy_param.device)
        points = points / math.sqrt(n)
        return points, predicted_n, predicted_formula


class FirstKSetGenerator(SetGenerator):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.points = nn.Parameter(torch.randn(cfg.max_n, cfg.set_channels).float())

    def forward(self, latent: Tensor, n: int = None, extrapolation=False):
        batch_size = latent.shape[0]
        n, predicted_n, predicted_formula = super().forward(latent, n, extrapolation)

        points = self.points[:n].unsqueeze(0).expand(batch_size, -1, -1)
        return points, predicted_n, predicted_formula


class TopNGenerator(SetGenerator):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.set_channels = cfg.set_channels
        self.cosine_channels = cfg.cosine_channels
        self.points = nn.Parameter(torch.randn(cfg.max_n, cfg.set_channels).float())

        angles = torch.randn(cfg.max_n, cfg.cosine_channels).float()
        angles = angles / (torch.norm(angles, dim=1)[:, None] + 1e-5)
        self.angles_params = nn.Parameter(angles)

        self.angle_mlp = MLP(cfg.latent_dim, self.cosine_channels, 32, 2)

        if self.predict_molecular_formula:
            self.mlp2 = MLP(self.cosine_channels, cfg.num_atom_types, cfg.hidden, cfg.num_mlp_layers)

        self.lin1 = nn.Linear(1, cfg.set_channels)
        self.lin2 = nn.Linear(1, cfg.set_channels)

    def forward(self, latent: Tensor, n: int = None, extrapolation=False):
        """ latent: batch_size x d
            self.points: max_points x d"""
        batch_size = latent.shape[0]
        n, predicted_n, predicted_formula = super().forward(latent, n, extrapolation)

        angles = self.angle_mlp(latent)
        angles = angles / (torch.norm(angles, dim=1)[:, None] + 1e-5)

        cosine = (self.angles_params[None, ...] @ angles[:, :, None]).squeeze(dim=2)
        cosine = torch.softmax(cosine, dim=1)
        # cosine = cosine / (torch.norm(set_angles, dim=1)[None, ...] + 1)        # 1 is here to avoid instabilities
        # Shape of cosine: bs x max_points
        srted, indices = torch.topk(cosine, n, dim=1, largest=True, sorted=True)  # bs x n

        indices = indices[:, :, None].expand(-1, -1, self.points.shape[-1])  # bs, n, set_c
        batched_points = self.points[None, :].expand(batch_size, -1, -1)  # bs, n_max, set_c

        selected_points = torch.gather(batched_points, dim=1, index=indices)

        alpha = self.lin1(selected_points.shape[1] * srted[:, :, None])
        beta = self.lin2(selected_points.shape[1] * srted[:, :, None])
        modulated = alpha * selected_points + beta
        return modulated, predicted_n, predicted_formula

