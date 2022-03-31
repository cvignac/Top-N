import sklearn.preprocessing
from rdkit import Chem
from torch_geometric.loader import DataLoader as TGDataLoader
from torch_geometric.datasets import QM9
from torch_geometric.data import Batch, Dataset
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, SequentialSampler
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_geometric.data import Data
import numpy as np


class CustomQM9(Dataset):
    def __init__(self, root, pre_transform=None, pre_filter=None):
        qm9 = QM9(root, transform=None, pre_transform=pre_transform, pre_filter=pre_filter)
        data_list = [qm9[i] for i in range(len(qm9))]
        lengths = [data.x.shape[0] for data in data_list]
        argsort = np.argsort(lengths)
        print("Reorganizing QM9 by molecule size...")
        self.data_list = [data_list[i] for i in argsort]  # List of tensors of increasing size
        self.split_indices = np.unique(np.sort(lengths), return_index=True)[1][1:]
        print("Done.")
        self.atom_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F'}       #  Warning: hydrogens have been removed
        self.bond_dict = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data_list[idx]
        # if self.transform:
        #     sample = self.transform(sample)
        return sample


class CustomGraphBatchSampler(BatchSampler):
    r""" Creates batches where all sets have the same size
    """
    def __init__(self, sampler, batch_size, drop_last, split_indices):
        super().__init__(sampler, batch_size, drop_last)
        self.split_indices = split_indices

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size or idx + 1 in self.split_indices:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        count = 0
        batch = 0
        for idx in self.sampler:
            batch += 1
            if batch == self.batch_size or idx + 1 in self.split_indices:
                count += 1
                batch = 0
        if batch > 0 and not self.drop_last:
            count += 1
        return count


class CustomDataLoader(TGDataLoader):
    def __init__(self, dataset, batch_size, drop_last=False):
        data_source = dataset
        sampler = SequentialSampler(data_source)
        batch_sampler = CustomGraphBatchSampler(sampler, batch_size, drop_last, dataset.split_indices)
        super().__init__(dataset, batch_sampler=batch_sampler)


class RemoveHydrogens(object):
    def __init__(self):
        pass

    def __call__(self, data):
        del data.pos
        del data.z
        del data.y
        E = to_dense_adj(data.edge_index, edge_attr=data.edge_attr, max_num_nodes=data.x.shape[0])  # 1, n, n, e_types



        non_hydrogens = data.x[:, 0] == 0
        hydrogens_ids = torch.nonzero(data.x[:, 0])
        data.x = data.x[non_hydrogens, 1:5]

        mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        E = E.squeeze(0)
        E = E[non_hydrogens, :, :]
        E = E[:, non_hydrogens, :]      # N, N, e_types

        A = torch.sum(E * torch.arange(1, E.shape[-1] + 1)[None,  None, :], dim=2)


        data.edge_index, edge_attr = dense_to_sparse(A)


        edge_attr = edge_attr.long().unsqueeze(-1) - 1

        data.edge_attr = torch.zeros(edge_attr.shape[0], 3)
        data.edge_attr.scatter_(1, edge_attr, 1)

        # data.edge_attr = data.edge_attr - 1

        # data.edge_attr = torch.hstack([E[data.edge_index[0], data.edge_index[1], :].unsqueeze(0)])

        # data.edge_attr = E[data.edge_index]
        #
        # for h in hydrogens_ids:
        #     mask += data.edge_index[0] == h
        #     mask += data.edge_index[1] == h
        #
        # data.edge_index = data.edge_index[:, ~mask]
        # data.edge_attr = data.edge_attr[~mask]
        # data.edge_attr = data.edge_attr[:, :-1]     # Aromatic bonds are not used in this dataset
        if data.edge_index.numel() > 0:
            assert data.edge_index.max() < len(data.x), f"{data.x}, {data.edge_index}"
        return data


def mol1_dataset():
    A = torch.zeros(3, 3)
    edges = [[0, 1, 1], [1, 0, 1], [1, 2, 2], [2, 1, 2]]
    for key in edges:
        A[key[0], key[1]] = key[2]
    edge_index, edge_attr1d = dense_to_sparse(A)
    edges = edge_index.shape[1]
    edge_attr = torch.zeros(edges, 3)
    for i, e in enumerate(edge_attr1d.long()):
        edge_attr[i, e - 1] = 1

    x = torch.zeros(3, 4)
    x[:, 0] = 1
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, idx=torch.arange(1))
    return [data]


def benzene_dataset():
    A = torch.zeros(6, 6)
    edges = [[0, 1, 1], [0, 5, 2], [1, 0, 1], [1, 2, 2], [2, 1, 2], [2, 3, 1], [3, 2, 1], [3, 4, 2], [4, 3, 2], [4, 5, 1],
             [5, 4, 1], [5, 0, 2]]

    for key in edges:
        A[key[0], key[1]] = key[2]

    edge_index, edge_attr1d = dense_to_sparse(A)
    edges = edge_index.shape[1]
    edge_attr = torch.zeros(edges, 3)
    for i, e in enumerate(edge_attr1d.long()):
        edge_attr[i, e - 1] = 1


    x = torch.zeros(6, 4)
    x[:, 0] = 1

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, idx=torch.arange(1))

    return [data]



class RandomMaskSelector(object):
    def __init__(self):
        """ Use as a Transform (not a pre_transform) in a pytorch geometric dataset """
        # TODO: set the initial numpy seed in the main script
        pass

    def __call__(self, data):

        # Select one mask
        num_masks = data.all_masks.shape[1]
        mask_id = int(np.random.uniform(0, num_masks, 1))
        mask = data.all_masks[mask_id]

        # Mask the attributes
        # Use the subgraph function in pytorch_geometric.utils
        new_data = None
        return new_data
