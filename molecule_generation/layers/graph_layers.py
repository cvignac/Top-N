import torch_geometric.nn as tgnn
from torch import Tensor
import torch


class EdgeTypesCounter(tgnn.MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, n, edge_index: Tensor, edge_attr: Tensor):
        """ edge_attr: num_edges x 3"""
        x = torch.zeros(n, edge_attr.shape[1], device=edge_index.device)
        out = self.propagate(edge_index, edge_attr=edge_attr, x=x)
        return out

    def message(self, x_j, edge_attr: Tensor):
        return edge_attr
