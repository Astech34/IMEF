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


# function to clean data
def sweep(var, threshold=1000, cart=True, newtype=np.nan):

    # extract x,y,z components if cart is true
    if cart == True:
        x_comp = var.loc[:, "x"].values  # x-component
        y_comp = var.loc[:, "y"].values  # y-component
        z_comp = var.loc[:, "z"].values  # z-component

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
        name = var.name  # get variable name
        cart_check = (
            True if "cart" in var.dims else False
        )  # check if coords need to be extrapolated

        if cart_check == True:
            for axi in ["x", "y", "z"]:
                # append var coordinates to df
                df[name + "_" + axi.upper()] = var.loc[:, axi].values
        else:
            # append var to df
            df[name] = var.values

    return df

def main():
    # LOCATE AND PULL MMS DATA

    # (*) glob module finds all pathnames given user-perscribed rules
    data_floc = "../mms-data/"  # directory for data file location
    dfile_mms1 = glob.glob(data_floc + "mms1*.nc")  # mms1 datafiles
    sample = "mms-data/mms1_imef_srvy_l2_5sec_20150901000000_20190101000000.nc"

    # open multiple files and load them along the time dimension
    #   - cache=False is required for lazy loading according to https://stackoverflow.com/a/45644654
    #   - decode_times=False is required for lazy loading according to https://stackoverflow.com/a/71158553
    #   - requires dask

    # open and combine datasets between 2015 to 2022 (mms1)
    #   - combine='nested' option indicates that the datasets should be combined along a new dimension
    #   - concat_dim='time' specifies the name of the new dimension
    mms_dataset = xr.open_mfdataset(
        dfile_mms1,
        combine="nested",
        concat_dim="time",
    )

    instr = "E_DIS"  # instrument to bin
    coord = "x"  # cooridnate
    mlt_ang_adj = 1.0 / 24.0 * (2.0 * np.pi)  # MLT conversion

    time = mms_dataset["time"].values  # time data
    dis_dat = mms_dataset["E_DIS"].loc[:, coord].values  # pull EF data (DIS)
    edi_dat = mms_dataset["E_EDI"].loc[:, coord].values  # pull EF data (EDI)

    # write xarray data to pandas df
    mms_df = write_toDf(
        mms_dataset["time"],
        mms_dataset["Dst"],
        mms_dataset["L"],
        mms_dataset["MLAT"],
        mms_dataset["Scpot"],
        mms_dataset["E_DIS"],
        mms_dataset["E_EDI"],
    )

    # combine datasets; replace NaN EDI data with DIS data, if exists
    coords = ["X", "Y", "Z"]
    for xyz in coords:
        mms_df["E" + xyz] = mms_df["E_EDI_" + xyz].combine_first(mms_df["E_DIS_" + xyz])

    # CLEANUP

    # average data into 1 min. bins
    mms_df = mms_df.resample(rule="1Min", on="time").mean()

    # # drop last row to account for OMNI dataset
    mms_df = mms_df[:-1]

    # # check
    # mms_df.head(5)

    # GET OMNIWEB DATA
    # (using hapi)
    from hapiclient import hapi

    # data server
    server = "https://cdaweb.gsfc.nasa.gov/hapi"

    # OMNI 1-minute dataset
    dataset = "OMNI_HRO2_1MIN"

    # pull start and stop times from MMS
    #   - done manually by copy-pasting index, will eventually want it to
    #       computationally pull time from the mms dataset
    #   - added extra minute so hapi times include final timestamp
    # combined_data.index[0]    >>Timestamp('2015-09-01 00:00:00')
    # combined_data.index[-1]   >>Timestamp('2022-08-31 23:59:00')

    t0 = "2015-09-01T00:00:00"
    tf = "2022-08-31T23:59:00"

    # The HAPI convention is that parameters is a comma-separated list. Here we request two parameters.
    omni_parameters = "IMF,Vx,Vy,Vz,SYM_H"

    # Configuration options for the hapi function.
    opts = {"logging": True, "usecache": True, "cachedir": "./hapicache"}

    # Get parameter data. See section 5 for for information on getting available datasets and parameters
    omni_data, meta = hapi(server, dataset, omni_parameters, t0, tf, **opts)

    # COMBINE DATA

    # create df to hold combined mms and omni datasets
    complete_df = mms_df.copy()

    # get var names from OMNI data in format that can be indexed
    omni_names = [s.strip() for s in omni_parameters.split(",")]

    # create omni df
    # omni_df = df = pd.DataFrame()

    # add time column
    # omni_df['time'] = omni_data['Time']

    # add variables
    for var in omni_names:
        complete_df["OMNI_" + var] = omni_data[var]

    # CLEAN DATA

    # interpolation (optional)
    # - (!) check - use datrange and go to first minute to last and generate all timestamps
    # - (!) show truly only interpolating up to 15
    complete_df = complete_df.interpolate(method="linear", limit=5)

    # drop rows with nans
    complete_df.dropna(axis=0, inplace=True)

    # return df, ready for training...
    # return complete_df

    # ...or, save it locally...
    floc = "prop_models/"
    complete_df.to_pickle(floc+"complete_training_df")


if __name__ == "__main__":
    main()
