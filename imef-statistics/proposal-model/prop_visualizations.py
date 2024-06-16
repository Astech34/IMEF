import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt, dates as mdates, ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy
from scipy.stats import binned_statistic
import prop_MCDO_functions as MCDO


def perfect_axis(ax, x, y):
    ''' create square plots with equal x/y axes'''

    plot_dim = max([max(x), max(y), abs(min(x)), abs(min(y))])
    print(f"dimension: {plot_dim}")
    ax.set_xlim([-plot_dim, plot_dim])
    ax.set_ylim([-plot_dim, plot_dim])
    ax.set_aspect("equal")


# lims = [
#     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
# ]

# # now plot both limits against eachother
# ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)


def perfect_axis_noNaN(ax, x, y):
    """create square plots with equal x/y axes, ignore nans"""

    plot_dim = max([np.nanmax(x), np.nanmax(y), abs(np.nanmin(x)), abs(np.nanmin(y))])
    print(f"dimension: {plot_dim}")
    ax.set_xlim([-plot_dim, plot_dim])
    ax.set_ylim([-plot_dim, plot_dim])
    ax.set_aspect("equal")


def plot_single_pred_index(target_name, actual_data, preds_data, scale=None):
    ''' 
    plot single targer over index for actual and predicted values; 
    scale variable allows to only plot within [i0:if]
    '''
    # [BI:06/13/2024]: eventually want to make scaling (zooming) optiona;
    
    indx_0 = scale[0]
    indx_f = scale[1]

    fig, ax = plt.subplots(1, 1)
    ax.plot(actual_data[indx_0:indx_f], alpha=0.5, label="actual")
    ax.plot(preds_data[indx_0:indx_f], c="k", label="prediction")
    ax.set_title(f"Actual vs. Prediction Values for Target: {target_name}")
    ax.set_xlabel(f"index count")
    ax.set_ylabel(f"{target_name}")
    ax.legend()


def plot_targets_overview(targets_lst, actual_data, preds_data):
    ''' plot subplots of actual data vs prediction over database index'''
    fig, axes = plt.subplots(len(targets_lst), 1)
    fig.tight_layout()
    for i in range(len(targets_lst)):
        axes[i].plot(actual_data[:, i], alpha=0.5, label="actual")
        axes[i].plot(preds_data[:, i], alpha=0.5, c="k", label="prediction")
        axes[i].set_ylabel(targets_lst[i])
        axes[i].legend(handlelength=0.1)
        axes[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))


def plot_preds_scatter(targets_lst, actual_data, preds_data, set_fig = True, axes=None, hex=True):
    
    # create figure if not established outside function
    if set_fig == True:
        fig, axes = plt.subplots(1, len(targets_lst), figsize=(9, 8))
        fig.tight_layout()

    for i in range(len(targets_lst)):
        px = preds_data[:, i]
        py = actual_data[:, i]

        # compute R, RMSE
        rmse = MCDO.rmse(px, py)
        ccoeff, pvalue = scipy.stats.pearsonr(px, py)

        # plot data
        ptxt = f"RMSE = {rmse:.2f} \n R = {ccoeff:.2f}"

        # if hex is true, plot as log count hexbins
        if hex == True:
            hexim = axes[i].hexbin(px,py,gridsize=(200,200),bins='log',cmap='GnBu_r')
        # otherwise, normal scatter plot
        else:
            axes[i].scatter(px, py, c="k", marker=".", s=15, alpha=1)

        # set axes attributes
        axes[i].set_title(f"Target: {targets_lst[i]}")
        axes[i].set_xlabel("predictions")
        axes[i].set_ylabel("true values")

        # polyfit line
        axes[i].plot(
            np.unique(px),
            np.poly1d(np.polyfit(px, py, 1))(np.unique(px)),
            c="r",
            linewidth=2,
        )

        # identity line (x=y)
        # (!) axes should be equal!
        axes[i].axline((0, 0), slope=1, c='r', linestyle='--',linewidth=1)

        # create equal axis
        perfect_axis(axes[i], px, py)

        # annotate metrics
        axes[i].annotate(
            ptxt,
            xy=(1, 0),
            xycoords="axes fraction",
            fontsize=10,
            horizontalalignment="right",
            verticalalignment="bottom",
        )
