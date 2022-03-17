import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f


class MultiHeadAttention(nn.Module):
    # MultiHeadAttention implementation based on "The Illustrated Transformer"
    def __init__(self, dimension: int, hidden_size: int, n_head=6):
        """
        Args:
            dimension: dimension of the set
            hidden_size: width of a head
            n_head: number of head
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.dim = dimension

        self.q_lin = nn.Linear(self.dim, self.hidden_size * self.n_head)
        self.k_lin = nn.Linear(self.dim, self.hidden_size * self.n_head)
        self.v_lin = nn.Linear(self.dim, self.hidden_size * self.n_head)

        self.out_layer = nn.Linear(self.hidden_size * self.n_head, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The considered set [batch size, size of set, dimension]

        Returns:
            Attention computed by the multi-head without any mask
        """
        batch_size = x.size()[0]
        p_per_set = x.size()[1]

        # Computation of value, query and key for attention
        v = self.v_lin(x).view(batch_size, p_per_set, self.n_head, self.hidden_size).transpose(1, 2)
        q = self.q_lin(x).view(batch_size, p_per_set, self.n_head, self.hidden_size).transpose(1, 2)
        k = self.k_lin(x).view(batch_size, p_per_set, self.n_head, self.hidden_size).transpose(1, 2).transpose(2, 3)

        # attention is computed according to : softmax(Q.K^T).V/d_k
        score = torch.matmul(q, k)
        score = score / np.sqrt(self.hidden_size)
        soft_score = f.softmax(score, dim=-1)

        attention = torch.matmul(soft_score, v)
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch_size, p_per_set, self.hidden_size * self.n_head)

        out = self.out_layer(attention)
        return out
