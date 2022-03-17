from rdkit import Chem
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
import yaml
import os.path as osp
from easydict import EasyDict
from torch import zeros
from torch_geometric.utils import to_dense_batch, to_dense_adj



def check_size(set1: Tensor, set2: Tensor):
    """ Args:
        set1: Tensor of a set [batch size, point per set, dimension]
        set2: Tensor of a set [batch size, point per set, dimension]
        both dimension must be equal
    Returns:
        The Chamfer distance between both sets
    """
    bs1, n, d1 = set1.size()
    bs2, m, d2 = set2.size()
    assert (d1 == d2 and bs1 == bs2)  # Both sets must live in the same space to be compared


def chamfer_loss(set1: Tensor, set2: Tensor) -> torch.Tensor:
        check_size(set1, set2)
        dist = torch.cdist(set1, set2, 2)
        out_dist, _ = torch.min(dist, dim=2)
        out_dist2, _ = torch.min(dist, dim=1)
        total_dist = (torch.mean(out_dist) + torch.mean(out_dist2)) / 2
        return total_dist


def hungarian_loss(set1, set2) -> torch.Tensor:
    check_size(set1, set2)
    batch_dist = torch.cdist(set1, set2, 2)
    numpy_batch_dist = batch_dist.detach().cpu().numpy()
    indices = map(linear_sum_assignment, numpy_batch_dist)
    loss = [dist[row_idx, col_idx].mean() for dist, (row_idx, col_idx) in zip(batch_dist, indices)]

    # Sum over the batch (not mean, which would reduce the importance of sets in big batches)
    total_loss = torch.sum(torch.stack(loss))
    return total_loss


class VAELoss(nn.Module):
    def __init__(self, loss, lmbdas: list, predict_atom_types: bool, predict_bond_types: bool, predict_n: bool):
        """  lmbda: penalization of the Gaussian prior on the latent space
             loss: any loss function between sets. """
        super().__init__()
        self.lmbdas = np.array(lmbdas)
        self.loss = loss
        self.predict_atom_types = predict_atom_types
        self.predict_bond_types = predict_bond_types
        self.predict_n = predict_n
        if predict_atom_types:
            self.atom_loss = torch.nn.CrossEntropyLoss(reduction='sum')
        if predict_bond_types:
            self.bond_loss = torch.nn.CrossEntropyLoss(reduction='sum')
        if self.predict_n:
            self.n_loss = torch.nn.MSELoss(reduction='sum')

    def forward(self, output: Tensor, mu: Tensor, log_var: Tensor, n: Tensor, real: Tensor):
        """
        Args:
            output: output of the network: [bs, n, 3]
            mu: mean computed in the network: [bs, latent_dim]
            log_var: log variance computed in the network: [bs, latent_dim]
            real: expected value to compare to the output: [bs, n, 3]
        Returns:
            The variational loss computed as the sum of the hungarian loss and the Kullback-Leiber divergence.
        """
        output_set, atom_types, bond_types = output
        device = output_set.device
        real_set, real_atom_types, real_bond_types = real
        real_n = float(real_set.shape[1]) * torch.ones(real_set.shape[0], dtype=torch.float32).to(device)
        # print(output_s    et.min().item(), output_set.max().item())
        reconstruction_loss = self.loss(output_set, real_set)
        dkl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Flatten the atom_types except along the last dimension
        num_atom_types = atom_types.shape[-1] if atom_types is not None else 1
        atom_types = atom_types.transpose(0, 2).reshape(num_atom_types, -1).transpose(0, 1)
        real_atom_types = torch.argmax(real_atom_types, dim=-1).flatten()

        atom_loss = self.atom_loss(atom_types, real_atom_types) if self.predict_atom_types else zeros(1).to(device)
        bond_loss = self.bond_loss(bond_types, real_bond_types) if self.predict_bond_types else zeros(1).to(device)
        n_loss = self.n_loss(n, real_n) if self.predict_n else zeros(1).to(device)
        total_loss = reconstruction_loss + \
                     self.lmbdas[0] * dkl + \
                     self.lmbdas[1] * atom_loss + \
                     self.lmbdas[2] * bond_loss + \
                     self.lmbdas[3] * n_loss

        return [total_loss, reconstruction_loss, self.lmbdas[0] * dkl, self.lmbdas[1] * atom_loss,
                self.lmbdas[2] * bond_loss, self.lmbdas[3] * n_loss]


class HungarianVAELoss(VAELoss):
    def __init__(self, lmbdas, use_atom_types, use_bond_types, predict_n):
        """  lmbda: penalization of the Gaussian prior on the latent space. """
        super().__init__(hungarian_loss, lmbdas, use_atom_types, use_bond_types, predict_n)


class ChamferVAELoss(VAELoss):
    def __init__(self, lmbdas, use_atom_types, use_bond_types, predict_n):
        super().__init__(chamfer_loss, lmbdas, use_atom_types, use_bond_types, predict_n)


class GraphMatching(object):
    def __init__(self, disable_matching: bool):
        self.disable_matching = disable_matching
        print("Matching  disabled?", disable_matching)

    def __call__(self, log_X_pred, A_pred, log_E_pred, X, A, E):
        """ A: adjacency matrix: bs x n x n  -- used to compute the matching
            E: edge_features: bs x n x n x num_edge_types
            X: atom types: bs x n x num_atom_types
            Assign a score to each atom base
        """
        if self.disable_matching:
            return log_X_pred, A_pred, log_E_pred, X, A, E

        bs, n = X.shape[0], X.shape[1]

        atom_types_mult = torch.arange(1, X.shape[-1] + 1).to(X.device)[None, None, :]

        real_atom_types = torch.sum(X * atom_types_mult, dim=2)    # bs, n
        if A is not None:
            edge_types_mult = torch.arange(1, E.shape[-1] + 1).to(X.device)[None, None, None, :]
            real_num_edges = torch.sum(A, dim=2)
            real_edge_types = torch.sum(E * edge_types_mult, dim=-1)    # bs x n x n
            neighbor_type = real_atom_types.unsqueeze(2).expand(bs, n, n).transpose(1, 2) * A
            neighbor_score = torch.sum(10 ** real_edge_types * neighbor_type, dim=2)
            real_score = real_atom_types * 1e5 + real_num_edges * 1e4 + neighbor_score
        else:
            real_score = real_atom_types

        # Warning: the predicted atom types are probabilities, not one hot encoding
        predicted_atom_types = torch.sum(torch.exp(log_X_pred) * atom_types_mult * 100, dim=2)  # bs, n
        if A is not None:
            predicted_num_edges = torch.sum(torch.sigmoid(A_pred), dim=2)
            predicted_edge_types = torch.sum(torch.exp(log_E_pred) * edge_types_mult, dim=-1)  # bs x n x n
            neighbor_type = predicted_atom_types.unsqueeze(2).expand(bs, n, n).transpose(1, 2) * torch.sigmoid(A_pred)
            neighbor_score = torch.sum(10 ** predicted_edge_types * neighbor_type, dim=2)
            predicted_score = predicted_atom_types * 1e5 + predicted_num_edges * 1e4 + neighbor_score
        else:
            predicted_score = predicted_atom_types

        srted_at_types, indices_real = torch.sort(real_score)
        srted_pred_at_types, indices_pred = torch.sort(predicted_score)        # bs x n

        perm_mat = log_X_pred.new_zeros(bs, n, n)
        pred_perm_mat = log_X_pred.new_zeros(bs, n, n)

        for i in range(bs):
            perm_mat[i, torch.arange(n), indices_real[i]] = 1
            pred_perm_mat[i, torch.arange(n), indices_pred[i]] = 1

        newX = perm_mat @ X
        newlogXpred = pred_perm_mat @ log_X_pred

        if A is not None and A_pred is not None:
            newA = (perm_mat @ A).transpose(1, 2)
            newA = (perm_mat @ newA).transpose(1, 2)

            newApred = (pred_perm_mat @ A_pred).transpose(1, 2)
            newApred = (pred_perm_mat @ newApred).transpose(1, 2)

            perm_mat = perm_mat.unsqueeze(1)
            pred_perm_mat = pred_perm_mat.unsqueeze(1)

            newE = (perm_mat @ E).transpose(1, 2)
            newE = (perm_mat @ newE).transpose(1, 2)

            newlogEpred = (pred_perm_mat @ log_E_pred).transpose(1, 2)
            newlogEpred = (pred_perm_mat @ newlogEpred).transpose(1, 2)
        else:
         newApred, newlogEpred, newA, newE = None, None, None, None

        return newlogXpred, newApred, newlogEpred, newX, newA, newE


class GraphVAELoss(nn.Module):
    def __init__(self, lmbdas: list, graph_matcher, predict_n: bool):
        """  lmbda: penalization of the Gaussian prior on the latent space
             loss: any loss function between sets. """
        super().__init__()
        self.lmbdas = np.array(lmbdas)
        self.graph_matcher = graph_matcher
        self.predict_n = predict_n

        self.atom_loss = torch.nn.NLLLoss(reduction='sum')
        # self.bond_loss = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.Tensor([0.3]))
        self.bond_types_loss = torch.nn.NLLLoss(reduction='sum')
        self.formula_loss = torch.nn.MSELoss(reduction='sum')
        self.num_edges_loss = torch.nn.MSELoss(reduction='sum')
        self.num_edge_types_loss = torch.nn.MSELoss(reduction='sum')
        self.atom_valency_loss = torch.nn.MSELoss(reduction='sum')

        if self.predict_n:
            self.n_loss = torch.nn.MSELoss(reduction='sum')

    def forward(self, output: Tensor, mu: Tensor, log_var: Tensor, n: Tensor, real):
        """
        Args:
            output: output of the network: X, A, E
            mu: mean computed in the network: [bs, latent_dim]
            log_var: log variance computed in the network: [bs, latent_dim]
            real: expected value to compare to the output: [bs, n, 3]
        Returns:
            The variational loss computed as the sum of the hungarian loss and the Kullback-Leiber divergence.
        """
        log_X_pred, A_pred, log_E_pred = output

        device = log_X_pred.device

        X, mask = to_dense_batch(real.x, real.batch)
        if real.edge_index.shape[1] > 0:
            E = to_dense_adj(real.edge_index, real.batch, real.edge_attr)
            A = (torch.sum(E, dim=-1) > 0).float()
        else:
            A = None
            E = None

        losses = []

        bs, n, _ = X.shape

        log_X_pred, A_pred, log_E_pred, X, A, E = self.graph_matcher(log_X_pred, A_pred, log_E_pred,
                                                                            X, A, E)
        real_n = float(X.shape[1]) * torch.ones(X.shape[0], dtype=torch.float32).to(device)

        losses.append(self.atom_loss(log_X_pred.transpose(1, 2), torch.argmax(X, dim=2)) / n)

        if A is not None:
            num_edge_types = E.shape[-1]
            diag = torch.eye(n, dtype=torch.bool)[None, :, :].expand(bs, -1, -1).to(A.device)
            losses.append(F.binary_cross_entropy(A_pred[~diag], A[~diag])
                          / (n * (n - 1)))

            A_mask = (A > 0)[..., None].expand(-1, -1, -1, num_edge_types)
            E_predictions = log_E_pred[A_mask].reshape(-1, num_edge_types)
            E_real = torch.argmax(E, dim=3)[A > 0]
            losses.append(self.bond_types_loss(E_predictions, E_real) / n)
        else:
            losses.append(0.0)
            losses.append(0.0)

        real_formula = torch.sum(X, dim=1)
        predicted_formula = torch.sum(torch.exp(log_X_pred), dim=1)
        losses.append(self.formula_loss(predicted_formula, real_formula) / n)

        if A is not None:
            real_num_edges = torch.sum(A, dim=(1, 2))
            predicted_num_edges = torch.sum(A_pred, dim=(1, 2))
            losses.append(self.num_edges_loss(predicted_num_edges, real_num_edges) / (n * (n - 1)))

            real_num_edge_types = torch.sum(E, dim=[1, 2])
            predicted_num_edge_types = torch.sum(torch.exp(log_E_pred), dim=(1, 2))
            losses.append(self.num_edge_types_loss(predicted_num_edge_types, real_num_edge_types) / n)

            valency_mult = torch.arange(1, 4).reshape(1, 1, 1, 3).to(output[0].device)
            predicted_valency = torch.sum(torch.exp(log_E_pred) * valency_mult * A_mask, dim=(2, 3))
            real_valency = torch.sum(E * valency_mult, dim=(2, 3))
            losses.append(self.atom_valency_loss(predicted_valency, real_valency) / (n * (n - 1)))
        else:
            for i in range(3):
                losses.append(0.0)

        dkl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        losses.append(dkl)

        n_loss = self.n_loss(n, real_n) if self.predict_n else zeros(1).to(device)

        losses.append(n_loss)

        losses = [lmbda * loss for lmbda, loss in zip(self.lmbdas, losses)]
        total_loss = sum(losses)
        losses.append(total_loss)

        # Compute extra metrics to better understand training performance
        X_accuracy = torch.sum(torch.argmax(log_X_pred, dim=2) == torch.argmax(X, dim=2)) / n
        if A is not None:
            A_accuracy = torch.sum((A_pred > 0.5).long() == (A > 0.5).long()) / n ** 2
            E_accuracy = torch.sum(torch.argmax(E_predictions, dim=1) == E_real)
            E_accuracy = E_accuracy * bs / torch.sum(A)         # Division by bs will happen later
        else:
            A_accuracy, E_accuracy = 1.0, 1.0

        extra_metrics = [X_accuracy, A_accuracy, E_accuracy]
        return losses, extra_metrics


class BasicMolecularMetrics(object):
    def __init__(self, atom_dict, bond_dict, dataset_smiles_list = None):
        self.atom_dict = atom_dict
        self.bond_dict = bond_dict
        self.dataset_smiles_list = dataset_smiles_list

    def build_molecule(self, X, A, E):
        assert len(X.shape) == 1
        assert len(A.shape) == 2
        assert len(E.shape) == 2
        mol = Chem.RWMol()
        for atom in X:
            a = Chem.Atom(self.atom_dict[atom.item()])
            mol.AddAtom(a)

        all_bonds = torch.nonzero(A)
        for bond in all_bonds:
            mol.AddBond(bond[0].item(), bond[1].item(), self.bond_dict[E[bond[0], bond[1]].item()])

        return mol

    def toSmiles(self, mol):
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return Chem.MolToSmiles(mol)

    def compute_validity(self, generated):
        """ generated: list of triplets (X, A, E)"""
        valid = []

        for graph in generated:
            mol = self.build_molecule(*graph)
            smiles = self.toSmiles(mol)
            if smiles is not None:
                valid.append(smiles)

        return valid, len(valid) / len(generated)

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate(self, generated):
        valid, validity = self.compute_validity(generated)
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_list is not None:
                novel, novelty = self.compute_novelty(unique)
                print(f"Uniqueness over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = 0.0
                novel = None
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None
            novel = None
        return [validity, uniqueness, novelty], unique
