import torch
import torch.nn as nn


class EdgePredictor(nn.Module):
    def __init__(self, dim_latent, num_atom_types, num_edge_types, hidden, hidden_final):
        self.latent_mlp = nn.Sequential(nn.Linear(dim_latent, hidden),
                                        nn.ReLU(),
                                        nn.Linear(hidden, hidden))
        self.final_mlp = nn.Sequential(nn.Linear(1 + num_atom_types + dim_latent, hidden_final),
                                       nn.ReLU(),
                                       nn.Linear(hidden_final, num_edge_types))

    def forward(self, latent, positions, atom_types):
        """ latent: bs x n x channels
            positions: bs x n x 3
            atom_types: bs x n x num_atom_types
            :returns edge_types: bs x n x n x num_edge_types"""
        latent = self.latent_mlp(latent)
        dist = torch.cdist(positions, positions).unsqueeze(-1)      # bs x n x n x 1
        atom_pairs = atom_types.unsqueeze(2) * atom_types.unsqueeze(1)  # bs x n x n x atom_types
        latent_pairs = latent.unsqueeze(2) * latent.unsqueeze(1)      # bs x n x n x hidden
        y = torch.cat((dist, atom_pairs, latent_pairs), dim=3)
        edge_types = self.final_mlp(y)
        return edge_types
