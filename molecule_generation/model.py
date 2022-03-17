import numpy as np

from set_generators import RandomSetGenerator, TopKSetGenerator, FirstKSetGenerator, MLPGenerator
from layers.attention import MultiHeadAttention
from layers.base_layers import MLP
from layers.aggregators import PNAAggregator, Set2Set
from layers.graph_layers import EdgeTypesCounter
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
import torch_geometric.nn as tgnn


def pick_set_generator(config):
    name = config.set_gen_name
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
        module = TransformerEncoderLayer(dim_in, cfg.n_heads_transformer, cfg.dim_feedforward_transformer)
    elif layer == 'Set2Set':
        module = Set2Set(dim_in, cfg.preprocessing_steps)
    elif layer == 'PNA':
        module = PNAAggregator(cfg.average_n)
    else:
        raise ValueError("Layer name not found.")
    return module


class SetEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim, hidden, self.latent_dim = cfg.in_dim, cfg.hidden, cfg.latent_dim
        self.initial_mlp = MLP(in_dim, hidden, cfg.hidden_initial, cfg.initial_mlp_layers, dim_in_2=cfg.num_atom_types)
        self.encoder_layers = nn.ModuleList()
        self.use_bn = cfg.use_bn
        if self.use_bn:
            self.bn_layers = nn.ModuleList()
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
        for i in range(len(self.encoder_layers)):
            out = F.relu(self.encoder_layers[i](x))              # [bs, n, d]
            if self.use_bn:
                out = self.bn_layers[i](out.transpose(1, 2)).transpose(1, 2)   # bs, n, hidden
            x = out + x if self.res else out

        z = self.pooling(x)
        z = self.final_mlp(z)
        mu = z[:, :self.latent_dim]
        log_var = z[:, self.latent_dim:]
        return mu, log_var


class SetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden, hidden_final = cfg.hidden_decoder, cfg.hidden_last_decoder
        self.use_bn = cfg.use_batch_norm
        self.res = cfg.use_residual
        self.cosine_channels = cfg.cosine_channels
        self.initial_mlp = MLP(cfg.set_channels,
                               cfg.hidden_decoder,
                               cfg.hidden_initial_decoder,
                               cfg.initial_mlp_layers_decoder,
                               skip=1, bias=True, dim_in_2=cfg.latent_dim - cfg.cosine_channels)

        self.decoder_layers = nn.ModuleList()
        if self.use_bn:
            self.bn_layers = nn.ModuleList()
        for layer in cfg.decoder_layers:
            self.decoder_layers.append(create_layer(layer, hidden, hidden, cfg))
            if self.use_bn:
                self.bn_layers.append(nn.BatchNorm1d(hidden))

    def forward(self, x, latent):
        """ x: batch_size, n, channels
            latent: batch_size, channels2. """
        x = F.relu(self.initial_mlp(x, latent[:, self.cosine_channels:].unsqueeze(1)))
        for i in range(len(self.decoder_layers)):
            out = F.relu(self.decoder_layers[i](x))              # [bs, n, d]
            if self.use_bn and type(x).__name__ != 'TransformerEncoderLayer':
                out = self.bn_layers[i](out.transpose(1, 2)).transpose(1, 2)  # bs, n, hidden
            x = out + x if self.res and type(x).__name__ != 'TransformerEncoderLayer' else out
        return x


class SetTransformerVae(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.encoder = SetEncoder(config.encoder_config)
        self.set_generator = pick_set_generator(config.set_generator_config)
        self.decoder = SetDecoder(config.decoder_config)
        self.normal = torch.distributions.normal.Normal(0.0, 1.0)

    def forward(self, x, atom_types, bond_types):
        """ x: bs, n, channels.
            atom_types: bs, n, num_atom_types
            bond_types: bs, n, n, num_bond_types."""
        n = x.shape[1]
        latent_mean, log_var = self.encoder(x, atom_types)
        latent_vector = self.reparameterize(latent_mean, log_var)
        x, predicted_n = self.set_generator(latent_vector, n)
        out = self.decoder(x, latent_vector)
        return [out, latent_mean, log_var, predicted_n]

    def generate(self, device):
        latent = self.normal.sample(torch.Size([self.latent_dim])).float().unsqueeze(0).to(device)
        x, _ = self.set_generator(latent)
        return self.decoder(x, latent)

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


class GraphEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_atom_types,  self.num_edge_types = config.num_atom_types, config.num_edge_types
        self.latent_dim = config.latent_dim
        self.hidden = config.hidden_encoder
        self.num_layers = config.num_layers_encoder
        self.max_num_nodes = config.max_num_nodes
        self.hidden_last = config.hidden_last_encoder
        self.use_batch_norm = config.use_batch_norm

        self.edge_counter = EdgeTypesCounter()
        self.initial_mlp = MLP(self.num_atom_types, self.hidden, self.hidden, config.initial_mlp_layers_encoder)

        self.graph_layers = nn.ModuleList(
            [tgnn.NNConv(self.hidden, self.hidden, nn.Linear(self.num_edge_types, self.hidden ** 2),
                         aggr='mean') for _ in range(self.num_layers)]
        )

        self.batch_norm_layers = nn.ModuleList(
            [tgnn.BatchNorm(self.hidden) for _ in range(self.num_layers)]
        )

        self.final_mlp = MLP(1 + self.num_edge_types + 3 * self.num_atom_types + 2 * self.hidden * self.num_layers,
                             2 * self.latent_dim, self.hidden_last, config.final_mlp_layers_encoder)

    def forward(self, data):
        """ x: N x n_atom_types
            edge_index: 2 x E (batched)
            edge_type:  E x num_edge_types.
            output: batch x """
        x, edge_index, edge_types, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_nodes = tgnn.global_add_pool(torch.ones(batch.shape[0], 1, device=x.device), batch)
        edge_counter = self.edge_counter(x.shape[0], edge_index, edge_types) / self.max_num_nodes

        edge_counter = tgnn.global_add_pool(edge_counter, batch)

        out = [torch.cat([num_nodes / self.max_num_nodes, edge_counter, tgnn.global_add_pool(x, batch) / self.max_num_nodes,
                          tgnn.global_mean_pool(x, batch), tgnn.global_max_pool(x, batch)], dim=1)]
        # channels in out: 1 + num_edge_types + 3 x num_atom_types

        x = self.initial_mlp(x)        # N_tot x hidden
        for layer, bn in zip(self.graph_layers, self.batch_norm_layers):
            new_x = layer(x, edge_index, edge_types)
            new_out = torch.cat([tgnn.global_mean_pool(new_x, batch),
                                 tgnn.global_max_pool(new_x, batch)], dim=1)
            out.append(new_out)
            x = torch.relu(new_x) + x
            if self.use_batch_norm:
                x = bn(x)

        out = torch.cat(out, dim=1)
        out = self.final_mlp(out)

        mu = out[:, :self.latent_dim]
        log_var = out[:, self.latent_dim:]
        return mu, log_var


class GraphDecoder(SetDecoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        hidden, hidden_final, atom_types = cfg.hidden_decoder, cfg.hidden_last_decoder, cfg.num_atom_types
        self.spatial_dim = cfg.spatial_dim
        self.last_set_layer = MLP(hidden, cfg.spatial_dim, hidden_final, cfg.final_mlp_layers_decoder)
        self.edge_type_mlp = MLP(2 * self.spatial_dim, cfg.num_edge_types + 1, cfg.hidden_edge_types,
                                 cfg.edge_types_layers)
        self.atom_type_layer = MLP(self.spatial_dim + cfg.num_edge_types, atom_types, hidden_final,
                                   cfg.atom_types_mlp_layers)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.use_bn = cfg.use_batch_norm
        self.bn_edge_p = nn.BatchNorm2d(2 * cfg.spatial_dim)
        self.bn_edge_type = nn.BatchNorm2d(3 + 2 * atom_types)

    def forward(self, x, latent):
        """ latent: bs x n x channels
            positions: bs x n x 3
            log_atom_types: bs x n x num_atom_types
            :returns edge_types: bs x n x n x num_edge_types"""
        num_atoms = x.shape[1]

        x = super().forward(x, latent)
        positions = self.last_set_layer(x)       # bs x n x spatial_dim

        positions_4d = positions.unsqueeze(2).expand(-1, -1, num_atoms, -1)

        edge_prob_input = torch.cat((positions_4d, positions_4d.transpose(1, 2)), dim=3)

        if self.use_bn:
            edge_prob_input = self.bn_edge_p(edge_prob_input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        edge_types_logits = self.edge_type_mlp(edge_prob_input)
        edge_types = self.log_softmax(edge_types_logits[..., :-1])

        edge_probs = 1 - self.softmax(edge_types_logits)[..., -1]

        sum_probs = torch.sum(edge_types, dim=2) / 9    # bs x n x 1
        atom_types_input = torch.cat((positions, sum_probs), dim=-1)
        log_atom_types = self.log_softmax(self.atom_type_layer(atom_types_input))

        return log_atom_types, edge_probs, edge_types


class GraphTransformerVae(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.cosine_channels = config.cosine_channels
        self.encoder = GraphEncoder(config)
        self.set_generator = pick_set_generator(config)
        self.decoder = GraphDecoder(config)
        self.normal = torch.distributions.normal.Normal(0.0, 1.0)

    def forward(self, data):
        n = int(len(data.batch) / len(data.idx))
        latent_mean, log_var = self.encoder(data)
        latent = self.reparameterize(latent_mean, log_var)
        x, predicted_n, predicted_formula = self.set_generator(latent, n)
        out = self.decoder(x, latent)
        return [out, latent_mean, log_var, predicted_n]

    def generate(self, device, print_generated: bool):
        latent = self.normal.sample(torch.Size([self.latent_dim])).float().unsqueeze(0).to(device)
        x, _, _ = self.set_generator(latent, n=None)
        log_atom_types, edge_probs, log_edge_types = self.decoder(x, latent)   # 1, N, atom_t - 1, N, N, 1 -- 1, N, N, e_type

        log_atom_types, edge_probs, log_edge_types = log_atom_types.squeeze(0), edge_probs.squeeze(0), log_edge_types.squeeze(0)

        if print_generated:
            np.set_printoptions(precision=3)
            print("Atoms", torch.exp(log_atom_types).cpu().detach().numpy())
            ep = edge_probs.cpu().detach().numpy()

            print("Mean:", ep.mean(), "Std:", ep.std(), "Full:", ep)
            et = torch.exp(log_edge_types).cpu().detach().numpy()
            print("Edge types: 0:", et[:, :, 0].mean(), et[:, :, 0].std(),
                  "1:", et[:, :, 1].mean(), et[:, :, 1].std(),
                  "2:", et[:, :, 2].mean(), et[:, :, 2].std())

        atoms = torch.multinomial(torch.exp(log_atom_types), num_samples=1).squeeze(1)

        edge_probs = torch.tril(edge_probs.squeeze(-1), diagonal=-1)
        A = torch.bernoulli(edge_probs).to(latent.device)
        E = torch.zeros(A.shape, dtype=torch.long).to(latent.device)
        E[A > 0] = torch.multinomial(torch.exp(log_edge_types[A > 0]), num_samples=1).squeeze(1).to(latent.device)

        return atoms, A, E

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
