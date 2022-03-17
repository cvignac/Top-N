import os
import os.path as osp

import numpy as np
import numpy.random as npr
import torch
import yaml
from easydict import EasyDict
import shutil
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import (BatchSampler, DataLoader, Dataset, Sampler,
                              SequentialSampler)
import argparse
from pathlib import Path

np.random.seed(0)
num_constraints = 3


def generate_one_point(config, previous):
    """ previous: N x 5 array
        Dimensions 0, 1, 2: spatial location - 3: current valency - 4: max valency. """
    valency = npr.choice(config.atoms_val, p=config.atom_probs)

    pos = config.min_bound + npr.uniform(size=3).astype(np.float32) * (config.max_bound - config.min_bound)
    if previous is None:     # The first point is always accepted
        return np.array([pos[0], pos[1], pos[2], 0, valency])[None, :], 0

    all_pos = previous[:, :3]
    dist_to_others = np.sqrt(np.sum((all_pos - pos) ** 2, axis=1))      # Size N

    min_dist = np.min(dist_to_others)
    if min_dist < config.min_dist:
        return None, 1

    if min_dist > config.neighbor_threshold:        # Atom is not connected
        return None, 2

    mask = dist_to_others < config.neighbor_threshold
    num_neighbors = np.sum(mask)

    # One atom should not have too many neighbors
    if num_neighbors > valency:
        return None, 3

    # The atom should not make other atoms have a valency too large
    if np.any(previous[:, -2] + mask > previous[:, -1]):
        return None, 3

    previous[:, -2] = previous[:, -2] + mask     # Update the valencies

    return np.vstack((previous, np.array([pos[0], pos[1], pos[2], num_neighbors, valency])[None, :])), 0   # 0: no error


def generate_one_set(config):
    """ Generate one set of points through rejection sampling."""
    n = config.n_max     # The final set may actually be smaller
    all_pos = None
    unsuccessful = 0
    statistics = np.zeros(4, dtype=int)
    while all_pos is None or all_pos.shape[0] < n:
        generated, error = generate_one_point(config, all_pos)
        if generated is None:
            statistics[error] += 1
            unsuccessful += 1
            if unsuccessful > config.try_before_stop and (all_pos.shape[0] > 4 or unsuccessful > 1000):
                break
        else:
            all_pos = generated
            unsuccessful = 0
            statistics[0] += 1

    positions = all_pos[:, :3]

    translation = np.random.randn(1, 3) * config.translation_std
    positions = positions + translation

    valencies = all_pos[:, 3].astype(int)
    one_hot_valencies = np.zeros((all_pos.shape[0], config.num_atom_types), dtype=np.float32)
    one_hot_valencies[np.arange(len(valencies)), valencies - 1] = 1

    all_info = np.hstack((positions, one_hot_valencies))
    return all_info, statistics


def generate_dataset(save_to: str):
    """ save_to: folder where to store the data. """
    # Root directory
    root_dir = Path(__file__).absolute().parents[1]
    save_dir = Path(save_to)
    # Config directory
    config_dir = root_dir.joinpath("config")
    yaml_file = config_dir.joinpath('config_synthetic.yaml')
    save_train_path = save_dir.joinpath('dataset_synthetic')
    save_test_path = save_dir.joinpath('dataset_synthetic_test')
    saved_config = save_dir.joinpath('dataset_synthetic.txt')
    save_statistics_path = save_dir.joinpath('valency_statistics')
    save_n_dist_path = save_dir.joinpath('n_statistics')

    with yaml_file.open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = EasyDict(config)
        config.min_bound = np.array(config.min_bound)
        config.max_bound = np.array(config.max_bound)
        print(config)

    all_sets = []
    global_statistics = np.zeros(num_constraints + 1)
    for i in range(config.n_train_graphs + config.n_test_graphs):
        all_points, statistics = generate_one_set(config)
        if all_points.shape[0] < 2:     # Require sets to have several points
            continue
        all_sets.append(all_points)
        global_statistics += statistics

    global_statistics = global_statistics.astype(np.float32) / config.n_train_graphs

    print("Global Statistics:")
    print(f"Average number of atoms  per graph: {global_statistics[0]:.2f}")
    print(f"Average number of atoms rejected because they were too close: {global_statistics[1]:.2f}")
    print(f"Average number of atoms rejected because they had no neighbor: {global_statistics[2]:.2f}")
    print(f"Average number of atoms rejected because valency is too large: {global_statistics[3]:.2f}")

    all_sets_with_batch = []

    n_dist = np.zeros(config.n_max + 1)

    for i, set in enumerate(all_sets):
        n_dist[set.shape[0]] += 1
        set = np.hstack((set, i * np.ones((set.shape[0], 1))))   # N_atoms_dataset x (3+atom_types+1): pos, val, set id
        all_sets_with_batch.append(set)

    all_train_sets_with_batch = all_sets_with_batch[:config.n_train_graphs]
    all_test_sets_with_batch = all_sets_with_batch[config.n_train_graphs:]
    train_dataset = np.vstack(all_train_sets_with_batch)
    test_dataset = np.vstack(all_test_sets_with_batch)

    atom_counts = np.array(np.sum(train_dataset[:, 3: -1], axis=0))
    atom_counts = atom_counts / np.sum(atom_counts)
    print(f"Distribution of atoms: {atom_counts}")

    n_dist = n_dist / np.sum(n_dist)
    print(f"Distribution of number of atoms: {n_dist}")

    if config.save:
        np.save(save_train_path, train_dataset)
        np.save(save_test_path, test_dataset)
        np.save(save_statistics_path, atom_counts)
        np.save(save_n_dist_path, n_dist)
        # Save the config in a file
        shutil.copy(yaml_file, saved_config)


class ConstrainedDataset(Dataset):
    def __init__(self, filepath, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.filepath = filepath
        self.transform = transform

        all_data = np.load(filepath)
        points = all_data[:, :-1].astype(np.float32)
        batch = all_data[:, -1]
        data_list = np.split(points, np.unique(batch, return_index=True)[1][1:])
        # Sort the data list by size
        lengths = [s.shape[0] for s in data_list]
        argsort = np.argsort(lengths)
        self.data_list = [data_list[i] for i in argsort]
        # Store indices where the size changes
        self.split_indices = np.unique(np.sort(lengths), return_index=True)[1][1:]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data_list[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class CustomBatchSampler(BatchSampler):
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


class SyntheticDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, drop_last=False):
        data_source = dataset
        sampler = SequentialSampler(data_source)
        batch_sampler = CustomBatchSampler(sampler, batch_size, drop_last, dataset.split_indices)
        super().__init__(dataset, batch_sampler=batch_sampler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-to', type=str, default='./data/')
    args = parser.parse_args()

    generate_dataset(args.save_to)
    # synth_dataset_path = DATA_DIR.joinpath('dataset_synthetic.npy')
    # dataset = ConstrainedDataset(synth_dataset_path)
    # loader = SyntheticDataLoader(dataset, batch_size=2)
    # for i, data in enumerate(loader):
    #     if i > 5:
    #         break
    #     print(data)
