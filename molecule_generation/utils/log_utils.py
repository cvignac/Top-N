import numpy as np
import torch


def log_train_metrics(args, losses, extra_metrics, lr, epoch, wandb, verbose=True):
    xt = extra_metrics
    print(f"Epoch {epoch}: Train loss: {losses[9]:.5f} | Atom probs: {losses[0]:.5f} | " +
           f"Edge probs: {losses[1]:.5f} | " +
          (f"Edge types {losses[2]:.5f} |" if losses[2] != 0 else '') +
          (f"Formula: {losses[3]:.5f} | " if losses[3] != 0 else '') +
          (f"Num edges: {losses[4]:.5f} | " if losses[4] != 0 else '') +
          (f"Num edge types: {losses[5]:.5f} | " if losses[5] != 0 else '') +
          (f"Train valency: {losses[6]:.5f} | " if losses[6] != 0 else '') +
          (f'KLe : {losses[7]:.5f} |' if losses[7] != 0 else '') +
          (f"Train N loss: {losses[8]:.5f} | " if losses[8] != 0 else '') +
          f"lr: {lr:.1e} | " +
          (f"Train X acc: {xt[0]:.5f} | " if xt[0] != 0 else '') +
          (f"Train A acc: {xt[1]:.5f} | " if xt[1] != 0 else '') +
          (f"Train E acc: {xt[2]:.5f} | " if xt[2] != 0 else ''))

    if args.wandb:
        dic = {name: losses[i] for i, name in enumerate(["Train Atom probs", "Train Edge probs",
                                                         "Train edge_types", "Train formula loss",
                                                         "Train num edges", "Train num edge types",
                                                         "Train valency", "KL", "Train N loss", "Train loss"])}
        dic['Train X Acc'] = xt[0]
        dic['Train A Acc'] = xt[1]
        dic['Train E Acc'] = xt[2]
        dic['lr'] = lr
        wandb.log(dic)


def log_evaluation_metrics(args, metrics, wandb):
    if args.wandb:
        wandb.log({"Validity": metrics[0], "Uniqueness": metrics[1],
                   "Novelty": metrics[2]})


def print_generated_mols(generated, num_graphs=20):
    for i, graph in enumerate(generated):
        if i >= num_graphs:
            break
        atoms, A, E = graph
        atoms = atoms.cpu().detach().numpy()
        print("Atoms", atoms)
        print(A.cpu().detach().numpy())
        print("Edge types", E[A.bool()].cpu().detach().numpy())
