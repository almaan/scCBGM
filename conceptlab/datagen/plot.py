from mpl_toolkits.axes_grid1 import make_axes_locatable

def matrix_plot(matrix, xlabel = None, ylabel = None, cmap = plt.cm.RdBu, cmap_size = 5):

    n_y,n_x = matrix.shape

    fig,ax  = plt.subplots(1,1)
    im =ax.imshow(matrix, cmap=cmap)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=f"{cmap_size}%", pad=0.05)

    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticks(np.arange(0.5, n_x + 0.5, 1), minor=True)
    y_ticks = np.arange(0.5, n_y)
    ax.set_yticks(y_ticks, minor = False)

    ax.grid(True, which='minor', color='black', linestyle='-', linewidth=1)
    ax.grid(True, which='major', color='black', linestyle='-', linewidth=1)
    ax.xaxis.grid(False, which='major')

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    #plt.colorbar(im,)
    plt.show()
