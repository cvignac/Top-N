import os
import numpy as np
import torch
import random
from utils.generate_synthetic_dataset import (
    ConstrainedDataset, SyntheticDataLoader)
from utils.metrics import (HungarianVAELoss, constrained_loss)
from model import SetTransformerVae
from utils.log_utils import log_train_metrics, log_test_metrics, log_evaluation_metrics
from utils.plot_utils import plot_reconstruction, plot_generated_sets
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
import numpy.random as npr
from ot.lp import emd, emd2

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)


def train(args, config, dataset_config, train_data_path, test_data_path, wandb):
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
    train_dataset = ConstrainedDataset(train_data_path)
    test_dataset = ConstrainedDataset(test_data_path)
    train_loader = SyntheticDataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = SyntheticDataLoader(test_dataset, batch_size=args.batch_size)

    # Compute the statistics of the number of nodes
    if config.set_generator_config.learn_from_latent:
        config.set_generator_config.n_prob = None
    else:
        c = Counter()
        for data in train_loader:
            num_sets, n = data.shape[0], data.shape[1]
            c[n] += num_sets
        n_max = max(c.keys())
        prob = torch.zeros(n_max + 1)
        for key, val in c.items():
            prob[key] = val
        prob = prob / torch.sum(prob)
        print(f"Probabilities for the number of nodes: {prob.cpu().detach().numpy() * 100}")
        config.set_generator_config.n_prob = prob



            # Define model, loss, optimizer
    model = SetTransformerVae(config).to(device)
    loss_fct = HungarianVAELoss(config.glob.lmbdas, config.predict_molecular_formula, config.glob.use_bond_types,
                                config.set_generator_config.learn_from_latent, dataset_config)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, min_lr=1e-6)
    # wandb.watch(model, log_freq=100)

    def train_epoch(epoch: int):
        model.train()
        losses = np.zeros(8)           # Train_loss, atom_loss, bond_loss, n_loss
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            data = [data[:, :, :3].contiguous(), data[:, :, 3:].contiguous(), None]

            output = model(*data)
            loss = loss_fct(*output, data)
            if torch.isnan(loss[0]):
                raise ValueError("Nan detected in the loss")
            loss[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            losses += [l.item() / len(train_loader.dataset) for l in loss]

        log_train_metrics(args, losses, optimizer.param_groups[0]['lr'], epoch, wandb, verbose=True)
        return losses

    def test():
        model.eval()
        with torch.no_grad():
            losses = np.zeros(8)           # Train_loss, atom_loss, bond_loss, n_loss
            for i, data in enumerate(test_loader):
                data = data.to(device)
                data = [data[:, :, :3].contiguous(), data[:, :, 3:].contiguous(), None]

                output = model.reconstruct(*data)
                loss = loss_fct(*output, data)
                losses += [l.item() / len(train_loader.dataset) for l in loss]

            log_test_metrics(args, losses, epoch, wandb, verbose=True)
            return losses


    def evaluate():
        """ Check the constraints on the generated dataset"""
        model.eval()
        with torch.no_grad():
            generated = [model.generate(device, extrapolation=False) for i in range(config.glob.n_eval)]
            losses = constrained_loss(generated, dataset_config)
            log_evaluation_metrics(args, losses, epoch, wandb, extrapolation=False)
        with torch.no_grad():
            generated = [model.generate(device, extrapolation=True) for i in range(config.glob.n_eval)]
            losses = constrained_loss(generated, dataset_config)
            log_evaluation_metrics(args, losses, epoch, wandb, extrapolation=True)

    # Train
    for epoch in range(0, args.epochs):
        if optimizer.param_groups[0]['lr'] < 1e-5:         # Stop training if learning rate is too small
            break
        losses = train_epoch(epoch)
        scheduler.step(losses[0])
        if epoch % args.evaluate_every == 0:
            test()
            evaluate()
        if args.plot_every > 0 and epoch % args.plot_every == 0:
            for i, data in enumerate(train_loader):
                data = data.to(device)
                data = [data[:, :, :3].contiguous(), None, None]
                output = model(*data)
                plot_reconstruction(f"rec_{args.name}_e{epoch}", output, data)
                if i > 0:
                    break
            with torch.no_grad():
                generated = [model.generate(device) for i in range(10)]
            plot_generated_sets(generated, f"gen_{args.name}_epoch{epoch}", num_prints=10, folder='./plots')

        if epoch % 1000 == 0:
            if hasattr(model.set_generator, "points"):
                np.set_printoptions(precision=3, suppress=True)
                print(model.set_generator.points.detach().cpu().numpy())
    evaluate()
    print("Training completed. Exiting.")

