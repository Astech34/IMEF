import numpy as np
from matplotlib import pyplot as plt
import xarray as xr


def draw_earth(ax):
    '''
    A handy function for drawing the Earth in a set of Polar Axes
    '''
    ax.fill_between(np.linspace(-np.pi / 2, np.pi / 2, 30), 0, np.ones(30), color='k')
    ax.plot(np.linspace(np.pi / 2, 3 * np.pi / 2, 30), np.ones(30), color='k')


def plot_data(nL, nMLT, dL, dMLT, imef_data):

    # Create a coordinate grid
    phi = (2 * np.pi * (imef_data['MLT'].values + dMLT/2) / 24).reshape(nL, nMLT)
    r = imef_data['L'].values.reshape(nL, nMLT) + dL/2
    Er = imef_data['E_mean'].loc[:, :, 'r'].values.reshape(nL, nMLT)
    Ephi = imef_data['E_mean'].loc[:, :, 'phi'].values.reshape(nL, nMLT)

    # Convert to cartesian coordinates
    # Scaling the vectors doesn't work correctly unless this is done.
    Ex = Er * np.cos(phi) - Ephi * np.sin(phi)
    Ey = Er * np.sin(phi) + Ephi * np.cos(phi)

    # Plot the data
    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, subplot_kw=dict(projection='polar'))

    # Plot the electric field
    # Scale makes the arrows smaller/larger. Bigger number = smaller arrows.
    # May need to be changed when more data points are present
    ax1 = axes[0, 0]
    ax1.quiver(phi, r, Ex, Ey, scale=10)
    ax1.set_xlabel("Electric Field")
    ax1.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
    ax1.set_theta_direction(1)

    draw_earth(ax1)

    # Plot the number of data points in each bin
    ax2 = axes[0, 1]
    ax2.set_xlabel("Count")
    im = ax2.pcolormesh(phi, r, imef_data['count'].data, cmap='YlOrRd', shading='auto')
    fig.colorbar(im, ax=ax2)

    plt.show()


def main():
    L_range = (0, 12)
    MLT_range = (0, 24)
    dL = 1  # RE
    dMLT = 1  # MLT (Does this catch the last bin values?)
    L = xr.DataArray(np.arange(L_range[0], L_range[1], dL), dims='L')
    MLT = xr.DataArray(np.arange(MLT_range[0], MLT_range[1], dMLT), dims='MLT')
    # Number of points in each coordinate
    nL = len(L)
    nMLT = len(MLT)

    imef_data = xr.open_dataset('binned.nc')
    plot_data(nL, nMLT, dL, dMLT, imef_data)


if __name__ == '__main__':
    main()