# imports
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import numpy as np
import pandas as pd
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import os

import prop_data_prep as pdp
import prop_nn_models as pnn
import prop_nn_functions as pnf
import prop_MCDO_functions as MCDO


# function to get indices of array that are non nans
def get_non_nan_inds(arr):
    return np.where( ~np.isnan(arr) )[0]


# save figure
def save_fig_loc(fname='fig', path=None, override=True):

    # file format
    ftype = ".pdf"

    # date string
    date_str = dt.today().strftime('%Y_%m_%d')

    fig_str = path + fname + "_" + date_str

    # if ovveride is false, create a new file with version flag
    if override == False:
        version = 1 # set version

        # check if file exists
        while os.path.isfile(fig_str + "_ver" + str(version)+ftype) == True:
            version +=1

        # update filename to include version number
        fig_str = fig_str +"_ver" + str(version)

    # save figure to designated path
    plt.savefig(fig_str + ftype, bbox_inches="tight", dpi=400, transparent=True)


import os

# Specify path
path = "/usr/local/bin/"

# Check whether the specified
# path exists or not
isExist = os.path.exists(path)
print(isExist)
