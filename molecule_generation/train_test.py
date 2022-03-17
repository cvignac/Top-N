import os
import numpy as np
import torch
from src.dataset_generation.generate_synthetic_dataset import (
    ConstrainedDataset, SyntheticDataLoader)
from src.paths import DATA_DIR
from utils.metrics import (HungarianVAELoss, constrained_loss)
from model import SetTransformerVae
from utils.log_utils import log_train_metrics, log_evaluation_metrics
from utils.plot_utils import plot_sets
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(args, config, dataset_config, wandb):
    use_cuda = args.gpu is not None and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda:" + str(args.gpu))
        torch.cuda.set_device(args.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    else:
        device = "cpu"
    args.device = device
    args.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print('Device used:', device)

    # Load dataset
    synth_dataset_path = DATA_DIR.joinpath('dataset_synthetic.npy')
    dataset = ConstrainedDataset(synth_dataset_path)
    loader = SyntheticDataLoader(dataset, batch_size=args.batch_size)

    # Define model, loss, optimizer
    model = SetTransformerVae(config).to(device)
    loss_fct = HungarianVAELoss(config.glob.lmbdas, config.glob.use_atom_types, config.glob.use_bond_types,
                                config.set_generator_config.learn_from_latent)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, min_lr=1e-6)
    # wandb.watch(model, log_freq=100)

    def train_epoch(epoch: int):
        model.train()
        losses = np.zeros(6)           # Train_loss, atom_loss, bond_loss, n_loss
        for i, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(device)
            data = [data[:, :, -3:].contiguous(), data[:, :, :-3].contiguous(), None]
            output = model(*data)
            loss = loss_fct(*output, data)
            loss[0].backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            losses += [l.item() / len(loader.dataset) for l in loss]

        log_train_metrics(args, losses, optimizer.param_groups[0]['lr'], epoch, wandb, verbose=True)
        return losses


    def evaluate():
        """ Check the constraints on the generated dataset"""
        model.eval()
        with torch.no_grad():
            generated = [model.generate(device) for i in range(config.glob.n_eval)]
            losses = constrained_loss(generated, dataset_config)
            log_evaluation_metrics(args, losses, epoch, wandb, verbose=True)

    # Train
    for epoch in range(0, args.epochs):
        if optimizer.param_groups[0]['lr'] < 1e-5:         # Stop training if learning rate is too small
            break
        losses = train_epoch(epoch)
        scheduler.step(losses[0])
        if epoch % args.evaluate_every == 0:
            evaluate()
        if args.plot_every > 0 and epoch % args.plot_every == 0:
            for i, data in enumerate(loader):
                data = data.to(device)
                data = [data, None, None]
                output = model(*data)
                plot_sets(output, data)
                if i > 1:
                    break
    evaluate()


