import matplotlib.pyplot as plt 
import scipy
from scipy import stats
import numpy as np
import pandas as pd
from datetime import datetime as dt
from sklearn.model_selection import train_test_split


# load and read datasets (MMS and OMNI)
d1 = '../mms-data/mms-jae/mms1_IonAndMag_NovDec2017.csv'
d2 = '../mms-data/mms-jae/OMNI_HRO_1MIN_1246701.csv'

df_mms=pd.read_csv(d1)
df_swi=pd.read_csv(d2)


# file cleanup for OMNI
swi_name = df_swi[df_swi.columns[0]].name # get time column name for SW data
df_swi = df_swi.rename(columns={swi_name: 'time'}) # replace name
df_swi['time']=pd.to_datetime(df_swi['time']) # set time values to datetime format
df_swi = df_swi[(df_swi['time'] > '2017-11-01 03:23:00') & (df_swi['time'] <= '2017-12-31 23:59:00')] # correct timeline with mms data

# file cleanup for MMS
df_mms['time']=pd.to_datetime(df_mms['time']) # recallibrate 'time' to datetime format
df_mms = df_mms.resample(rule='1Min', on='time').mean() # average data into 1 min. bins


# combine datasets into one df
df_swi = df_swi.reset_index(drop=True)
df_mms = df_mms.reset_index(drop=True)
df_ds = pd.concat([df_swi, df_mms], axis=1)

# interpolation (optional)
# - (!) check - use datrange and go to first minute to last and generate all timestamps
# - (!) show truly only interpolating up to 15
# df_ds = df_ds.interpolate(method='linear',limit=15)

# convert index to time index
df_ds.set_index('time', inplace=True)

# drop rows with nans
df_ds_nans = df_ds.copy()
df_ds.dropna(axis=0, inplace=True)


# define E = -v x B
vcols = ['VX','VY','VZ']
bcols = ['BX','BY','BZ']

# compute the cross product for each point using numpy.cross
df_ds[['EX','EY','EZ']] = -1 * np.cross(df_ds[vcols], df_ds[bcols])



# functions for data 

def prepare_3Darrs(input_arr, output_arr, lag=1, delay=1, next_steps=1):
    """
    Prepares input and output arrays into LSTM-acceptable format.
    (Thanks to help from https://tinyurl.com/3h7me36p and Andy)

    Args:
        input_arr (array): (1D/2D array with different time measurements as rows and features as columns)
        output_arr (array): (1D/2D array with different time measurements as rows and features as columns)
        lag (int, optional): Number of time steps to look back for creating samples. Defaults to 1.
        delay (int, optional): Number of time steps to look forward for creating labels. Defaults to 1.
        next_steps (int, optional): Number of future steps to predict. Defaults to 1.

    Retruns:
        Tuple containing prepared input and output arrays in 3D LSTM-acceptable format

    Example: 
        Predict the next density (n) from magnetic field Bz in 60 min. sequence:
        > prepare_3Darrs(B,n,lag=60,delay=1,next_steps=1)

    Notes:
        - assumes data is contiguous (no time gaps)
        - mms is binned in 1 min interval, so lag=60 is 1 hr.

    """

    # -----
    # From Andy: This code will take 2d arrays of time series X, Y and prepares them 
    # as 3d arrays for time series prediction with shape (num_sequences, sequence_lenth, num_features). 
    # Note that I didn't test (I don't think?) for next_steps and delay not equal to 1.

    # So if you wanted to predict the next density (n) from magnetic field (Bx, By, Bz all as B) 
    # using 60-minute sequences, then you'd call it like: 
    # > prepare_LSTM_3darrs(B,n,lag=60,delay=1,next_steps=1)

    # Also note that this assumes your data is contiguous (no time gaps!)
    # You could insert the missing times and interpolate them, then provide 
    #those results to this - or only provide the contiguous blocks. 
    #LSTMs can also be made to support predictions with missing data if you tweak 
    # it right, but that'll require extra work - probably best to not worry about that until after the proposal tho.
    # What mike does: rempve the data and jump to 45
    # -----

    # check input params
    if (next_steps < 1) or (lag < 1) or (delay < 1):
        raise ValueError('All autoregress params (lag, delay, and next_steps) '
                        + 'must be >= 1')
    
    # if given 1d arrs, turn into 2d arrs with dim(cols) = 1
    if len(input_arr.shape) == 1: input_arr = input_arr[:,np.newaxis] # (rows, cols)
    if len(output_arr.shape) == 1: output_arr = output_arr[:,np.newaxis]

    n_samples = input_arr.shape[0] # total no. of samples
    adj_samples = n_samples - lag - (delay - 1) - (next_steps - 1) # no. samples adjusted for lag, delay

    # create empty arrs to hold data
    prep_input = np.empty((adj_samples, lag, input_arr.shape[1]))
    prep_output = np.empty((adj_samples, next_steps, output_arr.shape[1]))

    # fill arrays with slices of the data
    for i in range(adj_samples):
        prep_input[i] = input_arr[i:i + lag] # input indices to use to build model/ prediction;
                                             # input shape (adj_samples, lag)
        prep_output[i] = output_arr[i + lag + delay - 1: i + lag + delay + next_steps - 1] # indices model will try and predict 
                                                                                           # output shape (adj_samples, next_steps)
        
    return prep_input, prep_output

   
