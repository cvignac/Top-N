import torch
import torch.nn as nn


class PNAAggregator(nn.Module):
    def __init__(self, average_n = None):
        super().__init__()
        average_n = 1 if average_n is None else average_n
        self.average_n = torch.Tensor([average_n])
        self.name = "PNA"
        self.dim_multiplier = 12    # The output has 12x more channels than the input

    def forward(self, x):
        """ x: batch_size x n x channels"""
        x = x.unsqueeze(-1)         # bs, n, c, 1
        n = torch.Tensor([x.shape[1]])
        assert n > 1
        # n = x.shape[1]
        scalers = torch.Tensor([torch.log(n + 1) / torch.log(self.average_n + 1),
                   torch.log(self.average_n + 1) / torch.log(n + 1)]).to(x.device)

        x = torch.cat((x, x * scalers[0], x * scalers[1]), dim=-1)

        aggregators = [torch.sum(x, dim=1), torch.max(x, dim=1)[0], torch.mean(x, dim=1),
                       torch.std(x, dim=1)]
        aggregators = [agg.unsqueeze(2) for agg in aggregators]
        z = torch.cat(aggregators, dim=2)       # bs, channels, 4, 3
        z = torch.reshape(z, (z.shape[0], -1))
        return z


class Set2Set(torch.nn.Module):
    r"""The global pooling operator based on iterative content-based attention
    from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Args:
        in_channels (int): Size of each input sample.
        processing_steps (int): Number of iterations :math:`T`.
        num_layers (int, optional): Number of recurrent layers, *.e.g*, setting
            :obj:`num_layers=2` would mean stacking two LSTMs together to form
            a stacked LSTM, with the second LSTM taking in outputs of the first
            LSTM and computing the final results. (default: :obj:`1`)
    """

    def __init__(self, in_channels, processing_steps, num_layers=1):
        super(Set2Set, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.dim_multiplier = 2
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        self.softmax = nn.Softmax(dim=0)
        self.lstm = torch.nn.LSTM(self.out_channels, self.in_channels,
                                  num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, x):
        """x: bs x n x channels. """
        # TODO: check the implementation
        batch_size = x.shape[0]

        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        q_star = x.new_zeros(1, batch_size, self.out_channels)

        x = x.transpose(0, 1)           # n, bs, hidden

        for i in range(self.processing_steps):
            q, h = self.lstm(q_star, h)    # q: 1, bs, hidden
            a = self.softmax(x * q)          # n, bs, hidden
            r = torch.sum(a * x, dim=0)     # bs, hidden
            q_star = torch.cat([q, r.unsqueeze(0)], dim=-1)
        return q_star.squeeze(0)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


if __name__ == '__main__':
    x = torch.randn((12, 20, 32))
    pna = PNAAggregator(average_n=19)
    set2set = Set2Set(32, 5, 1)

    z1 = pna(x)
    z2 = set2set(x)
    print(z1.shape, z2.shape)