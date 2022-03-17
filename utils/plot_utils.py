import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_generated_sets(generated, prefix, num_prints=10, folder='./plots'):
    for i, set in enumerate(generated):
        if i >= num_prints:
            break
        set = set[0].squeeze(0).detach().cpu().numpy()
        plt.clf()
        fig = plt.figure(0)
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        ax.scatter(set[:, 0], set[:, 1], set[:, 2], label='Generated')
        plt.legend()
        plt.savefig(f'{folder}/gen_{prefix}_{i}.png')


def plot_reconstruction(prefix, predictions, sources, num_prints=10, folder='./plots'):
    predictions = predictions[0][0]
    sources = sources[0]
    for i in range(min(num_prints, predictions.shape[0])):
        prediction = predictions[i].detach().cpu().numpy()       # Take only the points themselves
        source = sources[i].detach().cpu().numpy()

        fig = plt.figure(0)
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        # if len(prediction.shape) == 3:
        #     prediction = prediction[0]      # Take the first set of the batch
        # if len(source.shape) == 3:
        #     source = source[0]

        ax.scatter(prediction[:, 0], prediction[:, 1], prediction[:, 2], label='prediction')
        ax.scatter(source[:, 0], source[:, 1], source[:, 2], label='Source')
        plt.legend()

        plt.ion()
        plt.savefig(f'{folder}/gen_{prefix}_{i}.png')
