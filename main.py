import argparse
import os
import yaml
from easydict import EasyDict
from pathlib import Path
from utils.load_config import Configuration
from utils.generate_synthetic_dataset import generate_dataset
import pprint
import wandb
import numpy as np


def parse_args():
    """
    Parse args for the main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='Number of epoch', default=10000)
    parser.add_argument('--batch-size', type=int, help='Size of a batch', default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--gpu', type=int, help='Id of gpu device. By default use cpu')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--wandb', action='store_true', help="Use the weights and biases library")
    parser.add_argument('--name', type=str)
    parser.add_argument('--evaluate-every', type=int, default=100)
    parser.add_argument('--plot-every', type=int, default=-1)
    parser.add_argument('--factor', type=float, default=0.5, help="Learning rate decay for the scheduler")
    parser.add_argument('--patience', type=int, default=750, help="Scheduler patience")
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--runs', type=int, default=1, help="Number of runs to average")
    parser.add_argument('--generator', type=str, choices=['random', 'first', 'top', 'mlp'], default="top")
    parser.add_argument('--modulation', type=str, choices=['add', 'film'], default="film")
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    root_dir = Path(__file__).parent.resolve()
    # Config directory
    yaml_file = root_dir.joinpath("config", 'model_config.yaml')
    dataset_config_file = root_dir.joinpath("config", 'config_synthetic.yaml')
    train_data_path = Path(args.data_dir).joinpath('dataset_synthetic.npy')
    test_data_path = Path(args.data_dir).joinpath('dataset_synthetic_test.npy')
    val_statistics_path = Path(args.data_dir).joinpath('valency_statistics.npy')
    n_statistics_path = Path(args.data_dir).joinpath('n_statistics.npy')

    if not os.path.isfile(train_data_path) and os.path.isfile(test_data_path):
        print("Generating dataset...")
        generate_dataset(args.data_dir)
        print("Done.")

    f = open(Path(args.data_dir).joinpath('dataset_synthetic.txt'), "r")
    print(f.read())

    val_statistics = np.load(val_statistics_path)
    n_statistics = np.load(n_statistics_path)

    # Changes in CUDA_VISIBLE_DEVICES must be done before loading pytorch
    if type(args.gpu) == int:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        args.gpu = 0

    with yaml_file.open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = EasyDict(config)

    if args.generator == 'top':
        config["SetGenerator"]['name'] = "TopKGenerator"
    elif args.generator == 'first':
        config["SetGenerator"]['name'] = "FirstKGenerator"
    elif args.generator == 'random':
        config["SetGenerator"]['name'] = "RandomGenerator"
    elif args.generator == 'mlp':
        config["SetGenerator"]['name'] = "MLPGenerator"
    else:
        raise ValueError("Unknown generator")

    config['Decoder']['modulation'] = '+' if args.modulation == 'add' else 'film'

    with dataset_config_file.open() as f:
        dataset_config = yaml.load(f, Loader=yaml.FullLoader)
        dataset_config = EasyDict(dataset_config)

    dataset_config.val_statistics = val_statistics
    dataset_config.n_dist = n_statistics

    # Create a name for weights and biases
    if args.name:
        args.wandb = True
    if args.name is None:
        args.name = config["SetGenerator"]['name']

    # DO NOT MOVE THIS IMPORT
    import train_test
    pprint.pprint(config)

    wandb_config = config.copy()

    config["Global"]['num_atom_types'] = len(dataset_config.atom_probs)
    config["Global"]['dataset_max_n'] = dataset_config.n_max
    config = Configuration(config)

    for i in range(args.runs):
        wandb.init(project="set_gen", config=wandb_config, name=f"{args.name}_{i}",
                   settings=wandb.Settings(_disable_stats=True), reinit=True,
                   mode='online' if args.wandb else 'disabled')
        wandb.config.update(args)
        wandb.config.update(dataset_config)

        train_test.train(args, config, dataset_config, train_data_path, test_data_path, wandb)


if __name__ == '__main__':
    main()
