from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np


def matrix_plot(
    matrix: np.ndarray,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap=plt.cm.RdBu,
    cmap_size: int = 5,
    ax: plt.Axes | None = None,
        show: bool = True,
):

    n_y, n_x = matrix.shape

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    im = ax.imshow(matrix, cmap=cmap)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=f"{cmap_size}%", pad=0.05)

    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticks(np.arange(0.5, n_x + 0.5, 1), minor=True)
    y_ticks = np.arange(0.5, n_y)
    ax.set_yticks(y_ticks, minor=False)

    ax.grid(True, which="minor", color="black", linestyle="-", linewidth=1)
    ax.grid(True, which="major", color="black", linestyle="-", linewidth=1)
    ax.xaxis.grid(False, which="major")

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    # plt.colorbar(im,)
    if show:
        plt.show()
