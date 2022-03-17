from set_generators import RandomSetGenerator, TopKSetGenerator, FirstKSetGenerator, MLPGenerator
from utils.load_config import EncoderConfig, DecoderConfig
from layers.attention import MultiHeadAttention
from layers.base_layers import MLP
from layers.edge_predictor import EdgePredictor
from layers.aggregators import PNAAggregator, Set2Set
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F


def pick_set_generator(config):
    name = config.name
    print(name)
    if name == 'RandomGenerator':
        cls = RandomSetGenerator
    elif name == 'TopKGenerator':
        cls = TopKSetGenerator
    elif name == 'FirstKGenerator':
        cls = FirstKSetGenerator
    elif name =='MLPGenerator':
        cls = MLPGenerator
    else:
        raise ValueError("Generator not found")
    return cls(config)


def create_layer(layer, dim_in, dim_out, modules_config):
    cfg = modules_config
    if layer == "MultiHeadAttention":
        module = MultiHeadAttention(dim_in, cfg.head_width, cfg.n_heads)
    elif layer == "Linear":
        module = nn.Linear(dim_in, dim_out, bias=True)
    elif layer == "MLP":
        module = MLP(dim_in, dim_out, cfg.hidden_mlp, cfg.num_mlp_layers)
    elif layer == 'Transformer':
        module = TransformerEncoderLayer(dim_in, cfg.n_heads, cfg.dim_feedforward, dropout=0, batch_first=True)
    elif layer == 'Set2Set':
        module = Set2Set(dim_in, cfg.preprocessing_steps)
    elif layer == 'PNA':
        module = PNAAggregator(cfg.average_n)
    else:
        raise ValueError("Layer name not found.")
    return module


class CustomEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        in_dim, hidden, self.latent_dim = cfg.in_dim, cfg.hidden, cfg.latent_dim,
        self.initial_mlp = MLP(in_dim, hidden, cfg.hidden_initial, cfg.initial_mlp_layers, dim_in_2=cfg.num_atom_types)
        self.encoder_layers = nn.ModuleList()
        self.use_bn = cfg.use_bn
        if self.use_bn:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden)])
        self.res = cfg.use_residual
        modules_config = cfg.modules_config

        for layer in cfg.layers:
            module = create_layer(layer, hidden, hidden, modules_config)
            self.encoder_layers.append(module)
            if self.use_bn:
                self.bn_layers.append(nn.BatchNorm1d(hidden))

        self.pooling = create_layer(cfg.aggregator, hidden, -1, modules_config)
        aggregated_dim = self.pooling.dim_multiplier * hidden
        self.final_mlp = MLP(aggregated_dim, 2 * self.latent_dim, cfg.hidden_final, cfg.final_mlp_layers)

    def forward(self, x, atom_types):
        """ x (Tensor): batch_size x n x in_channels. """
        x = F.relu(self.initial_mlp(x, atom_types))
        if self.use_bn:
            x = self.bn_layers[0](x.transpose(1, 2)).transpose(1, 2)  # bs, n, hidden
        for i in range(len(self.encoder_layers)):
            out = F.relu(self.encoder_layers[i](x))              # [bs, n, d]
            if self.use_bn and type(x).__name__ != 'TransformerEncoderLayer':
                out = self.bn_layers[i + 1](out.transpose(1, 2)).transpose(1, 2)   # bs, n, hidden
            x = out + x if self.res and type(x).__name__ != 'TransformerEncoderLayer' else out
        z = F.relu(self.pooling(x))
        z = self.final_mlp(z)
        mu = z[:, :self.latent_dim]
        log_var = z[:, self.latent_dim:]
        return mu, log_var


class CustomDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        hidden, hidden_final = cfg.hidden, cfg.hidden_final
        modules_config = cfg.modules_config
        self.use_bn = cfg.use_bn
        self.res = cfg.use_residual
        self.initial_mlp = MLP(cfg.set_channels, cfg.hidden, cfg.hidden_initial,
                               cfg.initial_mlp_layers, skip=1, bias=True,
                               dim_in_2=cfg.latent_dim,
                               modulation=cfg.modulation)

        self.decoder_layers = nn.ModuleList()
        if self.use_bn:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden)])
        for layer in cfg.layers:
            self.decoder_layers.append(create_layer(layer, hidden, hidden, modules_config))
            if self.use_bn:
                self.bn_layers.append(nn.BatchNorm1d(hidden))

        self.final_mlp = MLP(hidden, 3, hidden_final, cfg.final_mlp_layers)

        self.use_bond_types = cfg.use_bond_types
        if self.use_bond_types:
            self.bond_type_layer = EdgePredictor(hidden_final, cfg.num_atom_types, cfg.num_bond_types,
                                                 hidden, cfg.hidden_final)

    def forward(self, x, latent):
        """ x: batch_size, n, channels
            latent: batch_size, channels2. """
        x = F.relu(self.initial_mlp(x, latent.unsqueeze(1)))

        if self.use_bn:
            x = self.bn_layers[0](x.transpose(1, 2)).transpose(1, 2)

        for i in range(len(self.decoder_layers)):
            out = F.relu(self.decoder_layers[i](x))              # [bs, n, d]
            if self.use_bn and type(x).__name__ != 'TransformerEncoderLayer':
                out = self.bn_layers[i + 1](out.transpose(1, 2)).transpose(1, 2)   # bs, n, hidden
            x = out + x if self.res and type(x).__name__ != 'TransformerEncoderLayer' else out

        return self.final_mlp(x)


class SetTransformerVae(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.encoder = CustomEncoder(config.encoder_config)
        self.set_generator = pick_set_generator(config.set_generator_config)
        self.decoder = CustomDecoder(config.decoder_config)
        self.normal = torch.distributions.normal.Normal(0.0, 1.0)

    def forward(self, x, atom_types, bond_types):
        """ x: bs, n, channels.
            atom_types: bs, n, num_atom_types
            bond_types: bs, n, n, num_bond_types."""
        n = x.shape[1]
        latent_mean, log_var = self.encoder(x, atom_types)
        latent_vector = self.reparameterize(latent_mean, log_var)
        x, predicted_n, predicted_formula = self.set_generator(latent_vector, n)
        out = self.decoder(x, latent_vector)
        out = [out, predicted_formula, None]
        return [out, latent_mean, log_var, predicted_n]

    def generate(self, device, extrapolation: bool):
        latent_vec = self.normal.sample(torch.Size([self.latent_dim])).float().unsqueeze(0).to(device)
        x, _, _ = self.set_generator(latent_vec, n=None, extrapolation=extrapolation)
        return self.decoder(x, latent_vec)

    def reconstruct(self, x, atom_types, bond_types):
        """ x: bs, n, channels.
            atom_types: bs, n, num_atom_types
            bond_types: bs, n, n, num_bond_types."""
        n = x.shape[1]
        latent_mean, log_var = self.encoder(x, atom_types)
        x, predicted_n, predicted_formula = self.set_generator(latent_mean, n)
        out = self.decoder(x, latent_mean)
        out = [out, predicted_formula, None]
        return [out, latent_mean, None, predicted_n]

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu: mean of the encoder's latent space
            log_var: log variance of the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(std) * std
        return z


