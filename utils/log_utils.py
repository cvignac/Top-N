

def log_train_metrics(args, losses, lr, epoch, wandb, verbose=True):
    if verbose:
        print(f"Epoch {epoch}: Train loss: {losses[0]:.5f} | Wasserstein: {losses[1]:.5f} | " +
               f"KL: {losses[2]:.5f} | " +
              (f"Formula: {losses[3]:.5f} |" if losses[3] != 0 else '') +
              (f'| Bond: {losses[4]:.5f} |' if losses[4] != 0 else '') +
              (f" N: {losses[5]:.5f} | " if losses[5] != 0 else '') +
              (f" NN: {losses[6]:.5f} | " if losses[6] != 0 else '') +
              (f" Val: {losses[7]:.5f} | " if losses[7] != 0 else '') +
              f"lr: {lr:.1e}")

    if args.wandb:
        dic = {name: losses[i] for i, name in enumerate(["Train loss", "Wasserstein loss", "KL loss",
                                                             "Formula loss", "Train bond loss", "Train N loss",
                                                         "Train NN loss", "Train Val loss"])}
        dic['lr'] = lr
        wandb.log(dic)



def log_test_metrics(args, losses, epoch, wandb, verbose=True):
    if verbose:
        print(f"Epoch {epoch}: Test loss: {losses[0]:.5f} | Test W2: {losses[1]:.5f} | " +
              (f" TestFormula: {losses[3]:.5f} |" if losses[3] != 0 else '') +
              (f'| Test Bond: {losses[4]:.5f} |' if losses[4] != 0 else '') +
              (f" Test N: {losses[5]:.5f} | " if losses[5] != 0 else '') +
              (f" Test NN: {losses[6]:.5f} | " if losses[6] != 0 else '') +
              (f" Test Val: {losses[7]:.5f} | " if losses[7] != 0 else ''))

    if args.wandb:
        dic = {name: losses[i] for i, name in enumerate(["Test loss", "Test W2",
                                                         "Test Formula loss", "Test bond loss", "Test N loss",
                                                         "Test NN loss", "Test Val loss"])}
        wandb.log(dic)


def log_evaluation_metrics(args, losses, epoch, wandb, extrapolation):
    ext = 'Extrapolation ' if extrapolation else ""
    print(f"Epoch {epoch}: {ext} Bounding box loss: {losses[0]:.4f} | NN loss: {losses[1]:.4f} |" +
          f" Valency dist: {losses[2]:.4f} |" +
          f" N dist: {losses[3]:.3f} |  Diversity score: {losses[4]:.3f}")
    if args.wandb:
        wandb.log({f"{ext}Bounding box loss": losses[0],
                   f"{ext}NN loss": losses[1],
                   f"{ext}Valency dist": losses[2],
                   f"{ext}N dist": losses[3],
                   f"{ext}Diversity score": losses[4]})

