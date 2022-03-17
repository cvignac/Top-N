import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_sets(prediction, source):

    prediction = prediction[0][0].detach().numpy()       # Take only the points themselves
    source = source[0][0].detach().numpy()

    fig = plt.figure(0)
    ax = Axes3D(fig)

    if len(prediction.shape) == 3:
        prediction = prediction[0]      # Take the first set of the batch
    if len(source.shape) == 3:
        source = source[0]

    ax.scatter(prediction[:, 0], prediction[:, 1], prediction[:, 2], label='prediction')
    ax.scatter(source[:, 0], source[:, 1], source[:, 2], label='Source')
    plt.legend()

    plt.ion()

    plt.draw()
    plt.pause(0.5)
    # input("Showing plot. Press [enter] to continue.")
    # plt.close()
    #
    # fig = plt.figure(1)
    # ax = Axes3D(fig)
    #
    # if len(prediction.shape) == 3:
    #     prediction = prediction[1]      # Take the first set of the batch
    # if len(source.shape) == 3:
    #     source = source[1]
    #
    # ax.scatter(prediction[:, 0], prediction[:, 1], prediction[:, 2], label='prediction')
    # ax.scatter(source[:, 0], source[:, 1], source[:, 2], label='Source')
    # plt.legend()
    #
    # plt.ion()
    #
    # plt.draw()
    # plt.pause(0.001)
    # input("Showing plots. Press [enter] to continue.")
    # plt.close()
