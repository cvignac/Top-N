import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Various Layers to build networks or initialize set


class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, width: int, nb_layers: int, skip=1, bias=True,
                 dim_in_2: int=None, modulation: str = '+'):
        """
        Args:
            dim_in: input dimension
            dim_out: output dimension
            width: hidden width
            nb_layers: number of layers
            skip: jump from residual connections
            bias: indicates presence of bias
            modulation (str): "+", "*" or "film". Used only if  dim_in_2 is not None (2 inputs to MLP)
        """
        super(MLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.width = width
        self.nb_layers = nb_layers
        self.modulation = modulation
        self.hidden = nn.ModuleList()
        self.lin1 = nn.Linear(self.dim_in, width, bias)
        if dim_in_2 is not None:
            self.lin2 = nn.Linear(dim_in_2, width)
            if modulation == 'film':
                self.lin3 = nn.Linear(dim_in_2, width)
        self.skip = skip
        self.residual_start = dim_in == width
        self.residual_end = dim_out == width
        for i in range(nb_layers-2):
            self.hidden.append(nn.Linear(width, width, bias))
        self.lin_final = nn.Linear(width, dim_out, bias)

    def forward(self, x: Tensor, y: Tensor=None):
        """
        MLP is overloaded to be able to take two arguments.
        This is used in the first layer of the decoder to merge the set and the latent vector
        Args:
            x: a tensor with last dimension equals to dim_in
        """
        out = self.lin1(x)
        if y is not None:
            out2 = self.lin2(y)
            if self.modulation == '+':
                out = out + out2
            elif self.modulation == '*':
                out = out * out2
            elif self.modulation == 'film':
                out3 = self.lin3(y)
                out = out * torch.sigmoid(out2) + out3
            else:
                raise ValueError(f"Unknown modulation parameter: {self.modulation}")
        out = F.relu(out) + (x if self.residual_start else 0)
        for layer in self.hidden:
            out = out + layer(F.relu(out))
        out = self.lin_final(F.relu(out)) + (out if self.residual_end else 0)
        return out


