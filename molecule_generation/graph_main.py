# from rdkit import Chem
import argparse
import os
import yaml
from easydict import EasyDict as edict
from src.paths import CONFIG_DIR
import wandb


def parse_args():
    """
    Parse args for the main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='Number of epoch', default=800)
    parser.add_argument('--batch-size', type=int, help='Size of a batch', default=512)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gpu', type=int, help='Id of gpu device. By default use cpu')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--wandb', action='store_true', help="Use the weights and biases library")
    parser.add_argument('--name', type=str)
    parser.add_argument('--evaluate-every', type=int, default=20)
    parser.add_argument('--plot-every', type=int, default=-1)
    parser.add_argument('--factor', type=float, default=0.5, help="Learning rate decay for the scheduler")
    parser.add_argument('--patience', type=int, default=200, help="Scheduler patience")
    parser.add_argument('--generator', type=str, choices=['random', 'first', 'top', 'mlp'], default="first")
    parser.add_argument('--modulation', type=str, choices=['add', 'film'], default="film")
    parser.add_argument('--print-graphs-every', type=int, default=-1)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    # Config files to import
    yaml_file = CONFIG_DIR.joinpath('config_graph.yaml')
    # dataset_config_file = CONFIG_DIR.joinpath('config_qm9.yaml')

    args = parse_args()
    # Changes in CUDA_VISIBLE_DEVICES must be done before loading pytorch
    if type(args.gpu) == int:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        args.gpu = 0

    with yaml_file.open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)

    if args.generator == 'top':
        config['set_gen_name'] = "TopKGenerator"
    elif args.generator == 'first':
        config['set_gen_name'] = "FirstKGenerator"
    elif args.generator == 'random':
        config['set_gen_name'] = "RandomGenerator"
        config['set_channels'] = 1
    elif args.generator == 'mlp':
        config['set_gen_name'] = 'MLPGenerator'
    else:
        raise ValueError("Unknown generator")

    print(config)
    wandb_config = config.copy()
    config = edict(config)

    # Create a name for weights and biases
    if args.name:
        args.wandb = True

    if args.name is None:
        args.name = 'VAE'

    for i in range(args.runs):
        wandb.init(project="graph_gen", config=wandb_config, name=args.name + str(i),
                   settings=wandb.Settings(_disable_stats=True),
                   mode='online' if args.wandb else 'disabled', reinit=True)
        wandb.config.update(args)

        # DO NOT MOVE THIS IMPORT
        import graph_train_test as train_test

        train_test.train(args, config, wandb)


if __name__ == '__main__':
    main()
