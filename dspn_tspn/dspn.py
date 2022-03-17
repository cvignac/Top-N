import torch
import torch.nn as nn
import torch.nn.functional as F


class DSPN(nn.Module):
    """ Deep Set Prediction Networks
    Yan Zhang, Jonathon Hare, Adam Prügel-Bennett
    NeurIPS 2019
    https://arxiv.org/abs/1906.06565
    """

    def __init__(self, encoder, set_channels, max_set_size, channels, iters, lr, generator):
        """
        encoder: Set encoder module that takes a set as input and returns a representation thereof.
            It should have a forward function that takes two arguments:
            - a set: FloatTensor of size (batch_size, input_channels, maximum_set_size). Each set
            should be padded to the same maximum size with 0s, even across batches.
            - a mask: FloatTensor of size (batch_size, maximum_set_size). This should take the value 1
            if the corresponding element is present and 0 if not.

        channels: Number of channels of the set to predict.

        max_set_size: Maximum size of the set.

        iter: Number of iterations to run the DSPN algorithm for.

        lr: Learning rate of inner gradient descent in DSPN.
        """
        super().__init__()
        self.encoder = encoder
        self.iters = iters
        self.lr = lr
        self.channels = channels
        self.generator = generator
        self.max_set_size = max_set_size
        self.set_channels = set_channels

        if generator == 'top':
            self.cosine_channels = 4
            self.starting_set = nn.Parameter(torch.rand(1, set_channels + self.cosine_channels, 2 * max_set_size))
            self.starting_mask = nn.Parameter(0.5 * torch.ones(1, 2 * max_set_size))
        elif generator == 'mlp':
            # TODO: adapt to the dataset
            self.gen_mlp = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, max_set_size * set_channels))
            self.starting_mask = nn.Parameter(0.5 * torch.ones(1, max_set_size))
        elif generator == 'random':
            self.starting_mask = nn.Parameter(0.5 * torch.ones(1, max_set_size))
        else:
            self.starting_set = nn.Parameter(torch.rand(1, set_channels, max_set_size))
            self.starting_mask = nn.Parameter(0.5 * torch.ones(1, max_set_size))

    def forward(self, target_repr):
        """
        Conceptually, DSPN simply turns the target_repr feature vector into a set.

        target_repr: Representation that the predicted set should match. FloatTensor of size (batch_size, repr_channels).
        Note that repr_channels can be different from self.channels.
        This can come from a set processed with the same encoder as self.encoder (auto-encoder), or a different
        input completely (normal supervised learning), such as an image encoded into a feature vector.
        """
        batch_size = target_repr.shape[0]

        if self.generator == 'top':
            latent_angle = target_repr[:, :self.cosine_channels]            # batch_size, cosine_channels
            set_angles = self.starting_set[:, :self.cosine_channels, :]     # 1, cosine_channels, all_points
            points = self.starting_set[:, self.cosine_channels:, :]         # 1, set_channels, all_points
            cosine = (set_angles.transpose(1, 2) @ latent_angle[:, :, None]).squeeze(dim=2)    # batch_size, all_points
            srted, indices = torch.topk(cosine, self.max_set_size, dim=1, largest=True, sorted=True)  # bs x n
            cosine_tilde = torch.sigmoid(srted)  # bs x n

            batched_mask = self.starting_mask.expand(batch_size, -1)  # batch_size, n_max
            selected_mask = torch.gather(batched_mask, dim=1, index=indices)

            indices = indices[:, None, :].expand(-1, points.shape[1], -1)  # bs, set_channels, n
            batched_points = points.expand(batch_size, -1, -1)             # bs, set_channels, n_max

            selected_points = torch.gather(batched_points, dim=2, index=indices)
            # selected_mask = torch.gather(batched_mask, dim=1, index=indices)
            modulated = selected_points * cosine_tilde[:, None, :]

            current_set = modulated
            current_mask = selected_mask
        elif self.generator == 'mlp':
            set_flat = self.gen_mlp(target_repr)
            current_set = set_flat.reshape(batch_size, self.set_channels, self.max_set_size)

            current_mask = self.starting_mask.expand(
                target_repr.size(0), self.starting_mask.size()[1]
            )
            # make sure mask is valid
            current_mask = current_mask.clamp(min=0, max=1)
        elif self.generator == 'random':
            current_set = torch.rand(target_repr.size(0), self.set_channels, self.max_set_size,
                                     requires_grad=True, device= self.starting_mask.device)
            current_mask = self.starting_mask.expand(
                target_repr.size(0), self.starting_mask.size()[1]
            )
            # make sure mask is valid
            current_mask = current_mask.clamp(min=0, max=1)
        else:
            # copy same initial set over batch
            current_set = self.starting_set.expand(
                target_repr.size(0), *self.starting_set.size()[1:]
            )
            current_mask = self.starting_mask.expand(
                target_repr.size(0), self.starting_mask.size()[1]
            )
            # make sure mask is valid
            current_mask = current_mask.clamp(min=0, max=1)

        # info used for loss computation
        intermediate_sets = [current_set]
        intermediate_masks = [current_mask]
        # info used for debugging
        repr_losses = []
        grad_norms = []

        # optimise repr_loss for fixed number of steps
        for i in range(self.iters):
            # regardless of grad setting in train or eval, each iteration requires torch.autograd.grad to be used
            with torch.enable_grad():
                if not self.training:
                    current_set.requires_grad = True
                    current_mask.requires_grad = True

                # compute representation of current set
                predicted_repr = self.encoder(current_set, current_mask)
                # how well does the representation matches the target
                repr_loss = F.smooth_l1_loss(
                    predicted_repr, target_repr, reduction="mean"
                )
                # change to make to set and masks to improve the representation
                set_grad, mask_grad = torch.autograd.grad(
                    inputs=[current_set, current_mask],
                    outputs=repr_loss,
                    only_inputs=True,
                    create_graph=True,
                )
            # update set with gradient descent
            current_set = current_set - self.lr * set_grad
            current_mask = current_mask - self.lr * mask_grad
            current_mask = current_mask.clamp(min=0, max=1)
            # save some memory in eval mode
            if not self.training:
                current_set = current_set.detach()
                current_mask = current_mask.detach()
                repr_loss = repr_loss.detach()
                set_grad = set_grad.detach()
                mask_grad = mask_grad.detach()
            # keep track of intermediates
            intermediate_sets.append(current_set)
            intermediate_masks.append(current_mask)
            repr_losses.append(repr_loss)
            grad_norms.append(set_grad.norm())

        return intermediate_sets, intermediate_masks, repr_losses, grad_norms
