import rdkit.Chem as Chem
import os
import numpy as np
import torch
from src.paths import DATA_DIR
from utils.metrics import GraphVAELoss, GraphMatching, BasicMolecularMetrics
from model import GraphTransformerVae
from utils.log_utils import log_train_metrics, log_evaluation_metrics, print_generated_mols
from utils.plot_utils import plot_sets
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.datasets import QM9
from utils.data_utils import CustomQM9, CustomDataLoader, RemoveHydrogens, benzene_dataset, mol1_dataset
from torch_geometric.nn import global_add_pool
from torch_geometric.data import DataLoader

qm9_atom_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F'}  # Warning: hydrogens have been removed
qm9_bond_dict = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]


def train(args, config,  wandb):
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
    remove_hydrogens = RemoveHydrogens()
    dataset = CustomQM9(DATA_DIR, pre_transform=remove_hydrogens)
    loader = CustomDataLoader(dataset, batch_size=args.batch_size)
    # dataset = benzene_dataset()
    # dataset = mol1_dataset()
    # loader = DataLoader(dataset)

    # Define model, loss, optimizer
    model = GraphTransformerVae(config).to(device)

    matcher = GraphMatching(disable_matching=False)
    loss_fct = GraphVAELoss(config.lambdas, matcher, config.predict_n)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, min_lr=1e-6)
    # wandb.watch(model, log_freq=100)

    molecular_metrics = BasicMolecularMetrics(qm9_atom_dict, qm9_bond_dict)


    def train_epoch(epoch: int):
        model.train()
        extra_metrics = np.zeros(3)
        losses = np.zeros(10)           # Train_loss, atom_loss, bond_loss, n_loss
        for i, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(device)
            output = model(data)    # [[atom_types, edge_p, edge_types], latent_mean, latent_var, predicted_n]
            loss, extra = loss_fct(*output, data)
            loss[-1].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0, error_if_nonfinite=True)
            optimizer.step()
            losses += [(l.item() if type(l) != float else l) / len(loader.dataset) for l in loss]   # number of graphs
            extra_metrics += [(m.item() if type(m) != float else m) / len(loader.dataset) for m in extra]

        log_train_metrics(args, losses, extra_metrics, optimizer.param_groups[0]['lr'], epoch, wandb, verbose=True)
        return losses

    def evaluate(print_generated:bool, num_prints=10):
        """ Check the constraints on the generated dataset"""
        model.eval()
        with torch.no_grad():
            generated = [model.generate(device, print_generated=print_generated * (i < num_prints))
                         for i in range(config.n_eval)]
            gen_metrics, unique = molecular_metrics.evaluate(generated)
            if unique is not None:
                print(unique)
            log_evaluation_metrics(args, gen_metrics, wandb)
            return generated


    # Train
    for epoch in range(0, args.epochs):
        if optimizer.param_groups[0]['lr'] < 1e-5:         # Stop training if learning rate is too small
            break
        losses = train_epoch(epoch)
        scheduler.step(losses[0])
        if epoch % args.evaluate_every == 0:
            generated = evaluate(print_generated=(epoch % args.print_graphs_every == 0), num_prints=10)
