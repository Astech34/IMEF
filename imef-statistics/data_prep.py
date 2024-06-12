# process mms datafiles
# imports
import matplotlib.pyplot as plt 
import scipy
from scipy import stats
import numpy as np
import pandas as pd
import netCDF4
import xarray as xr
import glob


# locate data
# (*) glob module finds all pathnames given user-perscribed rules
data_floc = "mms-data/"  # directory for data file location
dfile_mms1 = glob.glob(data_floc+"mms1*.nc") # mms1 datafiles
dfile_mms2 = glob.glob(data_floc+"mms2*.nc") # mms2 datafiles
sample = "mms-data/mms1_imef_srvy_l2_5sec_20150901000000_20190101000000.nc"

# open multiple files and load them along the time dimension
#   - cache=False is required for lazy loading according to https://stackoverflow.com/a/45644654
#   - decode_times=False is required for lazy loading according to https://stackoverflow.com/a/71158553
#   - requires dask

# open and combine datasets between 2015 to 2022 (mms1)
#   - combine='nested' option indicates that the datasets should be combined along a new dimension
#   - concat_dim='time' specifies the name of the new dimension
dat_mms1 = xr.open_mfdataset(dfile_mms1,
                             combine='nested', 
                             concat_dim='time',
)

# (old version) open and combine datasets between 2015 to 2022 (mms1)
# dat_mms1 = xr.open_mfdataset(dfile_mms1,
#     cache=False,
#     combine="by_coords",
#     coords=[
#         "time",
#     ],
#     decode_times=False,
# )


# open and combine datasets between 2015 to 2022 (mms2)
dat_mms2 = xr.open_mfdataset(dfile_mms2,
                             combine='nested', 
                             concat_dim='time',
)

# EDP coordinate fix
# (A/N): correct units - this may have been corrected?
dat_mms1['E_EDP']=dat_mms1['E_EDP'][:,:,0].drop(['cart',]).rename({'E_index': 'cart'}).assign_coords({'cart': ['x', 'y', 'z']}) 


# function to clean data
def sweep(var, threshold=1000, cart=True, newtype=np.nan):
    
    # extract x,y,z components if cart is true
    if cart==True:
        x_comp = var.loc[:,'x'].values # x-component
        y_comp = var.loc[:,'y'].values # y-component
        z_comp = var.loc[:,'z'].values # z-component

        # remove values above or eq. to threshold
        for comp in [x_comp, y_comp, z_comp]:
            comp[abs(comp) >= threshold] = newtype
    else:
        # remove values above or eq. to threshold
        for comp in var.values:
            comp[abs(comp) >= threshold] = newtype



# function to find and filter outliers
# (!) TBD

# function to write given E-Field as dataframe
def write_toDf(*args):

    # create mother df
    df = pd.DataFrame()

    for var in args:
        name = var.name # get variable name
        cart_check = True if 'cart' in var.dims else False # check if coords need to be extrapolated
        
        if cart_check == True:
            for axi in ['x','y','z']:
                 # append var coordinates to df
                 df[name+'_'+axi.upper()] = var.loc[:,axi].values
        else:
            # append var to df
            df[name] = var.values

    # ammend time?

    return df

# arrange training dataset
