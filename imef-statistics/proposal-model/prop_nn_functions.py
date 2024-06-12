import numpy as np

# import Neural_Networks as NN
import torch
import pandas as pd
import datetime as dt
import xarray as xr
from scipy.stats import binned_statistic_2d
from pathlib import Path
from warnings import warn
import os
import shutil
from urllib import request
from contextlib import closing
import requests
# from heliopy.data import omni  # no longeer supported !

from pymms.data import edi, util, fgm, fpi, edp
from pymms.sdc import mrmms_sdc_api as mms_api


#import imef.data.download_data as dd
import prop_data_manipulation as dm


# from torch.utils.data import DataLoader, TensorDataset

# %%%%%%%%%%


def random_undersampling(data, threshold=-40, quiet_storm_ratio=1.0):
    # Tbh not quite sure if I understood the algorithm correctly. It does what I think it wants, so I'll stick with it for now.
    # To be clear, this definitely undersamples the data. However the way I do it is by reducing the number of datapoints in each bin by a certain percentage. The authors may not do the same
    # But here is the link in case it's needed: https://link.springer.com/article/10.1007/s41060-017-0044-3

    intervals_of_storm_data = get_storm_intervals(
        data, threshold=threshold, max_hours=0
    )
    storm_counts = 0
    for start, end in intervals_of_storm_data:
        storm_counts += end - start
    if storm_counts == 0:
        print(
            Warning(
                "There is no storm data in the given dataset. Returning given data."
            )
        )
        return data
    quiet_counts = len(data["time"]) - storm_counts

    if quiet_counts > storm_counts:
        bins_to_undersample = reverse_bins(intervals_of_storm_data, len(data["time"]))
        percent_to_reduce = quiet_storm_ratio * storm_counts / quiet_counts

        if percent_to_reduce >= 1:
            raise ValueError(
                "quiet_storm_ratio is too large. The max value for this dataset is "
                + str(quiet_counts / storm_counts)
            )
        elif percent_to_reduce <= 0:
            raise ValueError(
                "quiet_storm ratio is too small. It must be greater than 0"
            )

        all_times = []
        for start, end in intervals_of_storm_data:
            all_times.append(data["time"].values[start:end])
        for start, end in bins_to_undersample:
            new_times_in_bin = np.random.choice(
                data["time"][start:end],
                int(percent_to_reduce * (end - start)),
                replace=False,
            )
            all_times.append(new_times_in_bin)
        all_times = np.concatenate(all_times)
        all_times = np.sort(all_times)

        undersampled_data = data.sel(time=all_times)

        return undersampled_data
    else:
        # I don't know if a) this will ever come up, or b) if this did come up, we would want to undersample the storm data. So raising error for now
        print(
            Warning("There is more storm data than quiet data. Skipping undersampling.")
        )
        return data


def get_NN_inputs(
    imef_data,
    remove_nan=True,
    get_target_data=True,
    use_values=["Kp"],
    usetorch=True,
    undersample=None,
):
    # This could be made way more efficient if I were to make the function not store all the data even if it isn't used. But for sake of understandability (which this has little of anyways)
    # the random_undersampling argument should be a float, if it is not none. The float represents the quiet_storm_ratio

    #
    if "Kp" in use_values:
        Kp_data = imef_data["Kp"]
    if "Dst" in use_values:
        Dst_data = imef_data["Dst"]
    if "Symh" in use_values:
        Symh_data = imef_data["Sym-H"]

    if remove_nan == True:
        imef_data = imef_data.where(
            np.isnan(imef_data["E_con"][:, 0]) == False, drop=True
        )
    if undersample != None and len(imef_data["time"].values) != 0:
        imef_data = random_undersampling(imef_data, quiet_storm_ratio=undersample)

    # Note that the first 5 hours of data cannot be used, since we do not have the preceding 5 hours of index data to get all the required data. Remove those 5 hours
    # Since there is now a try in the for loop, this shouldn't be needed. But just in case something strange comes up ill leave it here
    # imef_data = imef_data.where(imef_data['time']>=(imef_data['time'].values[0]+np.timedelta64(5, 'h')), drop=True)

    design_matrix_array = None
    if get_target_data == True:
        times_to_keep = []
    for counter in range(0, len(imef_data["time"].values)):
        new_data_line = []
        time_intervals = pd.date_range(
            end=imef_data["time"].values[counter], freq="5T", periods=60
        )
        try:
            if "Kp" in use_values:
                Kp_index_data = Kp_data.sel(time=time_intervals).values.tolist()
                new_data_line += Kp_index_data
            if "Dst" in use_values:
                Dst_index_data = Dst_data.sel(time=time_intervals).values.tolist()
                new_data_line += Dst_index_data
            if "Symh" in use_values:
                Symh_index_data = Symh_data.sel(time=time_intervals).values.tolist()
                new_data_line += Symh_index_data

            # Along with the indices, we include 3 extra values to train on: The distance from the Earth (L), cos(MLT), and sin(MLT)
            # the_rest_of_the_data = np.array([imef_data['L'].values[counter], np.cos(np.pi / 12 * imef_data['MLT'].values[counter]), np.sin(np.pi / 12 * imef_data['MLT'].values[counter])]).tolist()

            # Along with the indices, we include 5 extra values to train on: The distance from the Earth (L), cos(MLT), sin(MLT), cos(MLAT), and sin(MLAT)
            the_rest_of_the_data = np.array(
                [
                    imef_data["L"].values[counter],
                    np.cos(np.pi / 12 * imef_data["MLT"].values[counter]),
                    np.sin(np.pi / 12 * imef_data["MLT"].values[counter]),
                    np.cos(imef_data["MLAT"].values[counter]),
                    np.sin(imef_data["MLAT"].values[counter]),
                ]
            ).tolist()
            new_data_line += the_rest_of_the_data

            if design_matrix_array == None:
                design_matrix_array = [new_data_line]
            else:
                design_matrix_array.append(new_data_line)

            if get_target_data == True:
                times_to_keep.append(imef_data["time"].values[counter])
        except Exception as ex:
            # This should only be entered when there is not enough data to fully create the NN inputs required (aka previous 5 hours of index data, and location data)
            # print(ex)
            # raise ex
            pass

    if usetorch == True:
        design_matrix_array = torch.tensor(design_matrix_array)
    else:
        design_matrix_array = np.array(design_matrix_array)

    if get_target_data == True:
        efield_data = imef_data["E_con"].sel(time=times_to_keep).values

        if usetorch == True:
            efield_data = torch.from_numpy(efield_data)

        return design_matrix_array, efield_data
    else:
        return design_matrix_array


# %%%%%%%%%


def get_mec_data(sc, mode, level, ti, te, binned=False):
    # [BI:5/28/24] added some clarifying documentation, no major changes.
    """
    Load MEC data.
    Parameters
    ----------
    sc : str
        Spacecraft identifier: {'mms1', 'mms2', 'mms3', 'mms4'}
    mode : str
        Data rate mode: {'srvy', 'brst'}
    level : str
        Data level: {'l2'}
    ti, te : `datetime.datetime`
        Start and end of the data interval
    binned : bool
        Bin/average data into 5-minute intervals
    Returns
    -------
    mec_data : `xarray.Dataset`
        MEC ephemeris data
    """
    # binned=True bins the data into 5 minute bins in the intervals (00:00:00, 00:05:00, 00:10:00, etc)
    # For example, the bin 00:10:00 takes all the data from 00:07:30 and 00:12:30 and bins them
    # The first bin will not have enough data to bin into 5 minute intervals (It goes into the previous day).
    # But we also don't want values to overlap from day to day, so we have to take away another 2.5 minutes from the end so that we don't see repeats
    if binned == True:
        ti = ti - dt.timedelta(minutes=2.5)
        te = te - dt.timedelta(minutes=2.5)

        # the bin_5min program requires a multiple of 5 minutes from start to end (so no data is left off)
        # check if the time range given is a multiple of 5 minutes, and if it isn't, add onto the end whatever time is needed to make it a 5 minute interval
        if ((te - ti) % dt.timedelta(minutes=5)) / dt.timedelta(seconds=1) != 0:
            te = te + dt.timedelta(
                minutes=(
                    5
                    - (
                        (te - ti) / dt.timedelta(minutes=5)
                        - int((te - ti) / dt.timedelta(minutes=5))
                    )
                    * 5
                )
            )

    # The names of the variables that will be downloaded
    r_vname = "_".join((sc, "mec", "r", "gse"))
    v_vname = "_".join((sc, "mec", "v", "gse"))
    mlt_vname = "_".join((sc, "mec", "mlt"))
    mlat_vname = "_".join((sc, "mec", "mlat"))
    l_dip_vname = "_".join((sc, "mec", "l", "dipole"))

    # The names of the indices used for the radius (?) and velocity data
    r_lbl_vname = "_".join((sc, "mec", "r", "gse", "label"))
    v_lbl_vname = "_".join((sc, "mec", "v", "gse", "label"))

    # Get MEC data
    mec_data = util.load_data(
        sc,
        "mec",
        mode,
        level,
        optdesc="epht89d",
        start_date=ti,
        end_date=te,
        variables=[r_vname, v_vname, mlt_vname, l_dip_vname, mlat_vname],
    )

    # Rename variables
    mec_data = mec_data.rename(
        {
            r_vname: "R_sc",
            r_lbl_vname: "R_sc_index",
            v_vname: "V_sc",
            v_lbl_vname: "V_sc_index",
            mlt_vname: "MLT",
            l_dip_vname: "L",
            mlat_vname: "MLAT",
            "Epoch": "time",
        }
    )

    if binned == True:
        # [BI:5/28/24] (!) check: why no labels for L, MLT?
        mec_data = dm.bin_5min(
            mec_data, ["V_sc", "R_sc", "L", "MLT"], ["V_sc", "R_sc", "", ""], ti, te
        )

    return mec_data


def download_ftp_files(remote_location, local_location, fname_list):
    """
    Transfer files from FTP location to local location
    Parameters
    ----------
    remote_location : str
        Location on the FTP server where files are located
    local_location : str
        Local location where remote FTP files are to be copied
    fname_list : list of str
        List of files on the FTP site to be transferred
    """
    # this always downloads the file, in contrast to the other which only downloads it if the file doesn't already exist in data/kp
    # for fname in fname_list:
    #     # note this will redownload a file that has already been downloaded.
    #     with open(local_location + fname, 'wb') as f:
    #         with closing(request.urlopen(remote_location + fname)) as r:
    #             shutil.copyfileobj(r, f)

    # this does the same thing, but if the file name already exists, it doesn't download the data. note that this doesn't work flawlessly,
    # as any file that is downloaded from an incomplete month/year will not be updated, which can cause problems. while just deleting the file would fix it, meh
    for fname in fname_list:
        # Check if they exist
        if os.path.isfile(local_location + fname) == 0:
            # If they do not exist, create the new file and copy down the data
            # Note that this will not update existing files. May want to figure that out at some point
            with open(local_location + fname, "wb") as f:
                with closing(request.urlopen(remote_location + fname)) as r:
                    shutil.copyfileobj(r, f)


def read_txt_files(fname_list, local_location=None, mode="Kp"):
    # [BI:6/06/24] added some clarifying documentation, no major changes. Will probably need to review, but ok for now.
    """
    Reads data into a Pandas dataframe
    Parameters
    ----------
    fname_list : list of str
        Files containing Kp index
    local_location : str
        Path to where files are stored, if it isn't included in fname_list
    Returns
    -------
    full_data : `pandas.DataFrame`
        Kp data
    """
    # [BI:6/06/24] why 29? will need to look into this.
    if mode == "Kp":
        header = 29
        footer = 0
    if mode == "symh":
        header = 0
        footer = 0

    # Combine all of the needed files into one dataframe
    for fname in fname_list:
        if (
            mode == "Dst"
            and int(fname[13:17]) == 2021
            and int(fname[18:20]) >= 8
            or mode == "Dst"
            and int(fname[13:17]) >= 2022
        ):
            header = 34
            footer = 55
        elif mode == "Dst" and int(fname[13:17]) >= 2020:
            header = 34
            footer = 40
        elif mode == "Dst" and int(fname[13:17]) < 2020:
            header = 28
            footer = 41

        # Read file into a pandas dataframe, and remove the text at the top
        if local_location is not None:
            oneofthem = pd.read_table(
                local_location + fname, header=header, skipfooter=footer
            )
        else:
            oneofthem = pd.read_table(fname, header=header, skipfooter=footer)

        if fname == fname_list[0]:
            # If this is the first time going through the loop, designate the created dataframe as where all the data will go
            full_data = oneofthem
        else:
            # Otherwise, combine the new data with the existing data
            full_data = pd.concat([full_data, oneofthem], ignore_index=True)

    return full_data


# If you are having problems with this function, delete all Kp files in data/kp and run again. This may fix it # [BI:5/28/24] oh?
def get_kp_data_old(ti, te, expand=[None]):
    # location of the files on the server (GFZ Potsdam)
    remote_location = "ftp://ftp.gfz-potsdam.de/pub/home/obs/Kp_ap_Ap_SN_F107/"
    # local directory to hold data files
    local_location = "kp_data"  # [BI:5/28/24] changed to my local directory

    # parts of the filename, will be put together along with a year number. final product eg: Kp_ap_2018.txt
    file_name_template = "Kp_ap_"
    file_name_extension = ".txt"

    # Where the list of data points required will be stored
    fname_list = []
    increment = ti.year

    # Making the names of all the required files
    while increment <= te.year:
        fname_list.append(file_name_template + str(increment) + file_name_extension)
        increment += 1

    # # [BI:5/28/24] maybe write a check for the files?
    # # [BI:5/28/24] ok, download_ftp_files has the check built-in, maybe rewrite...
    # for fname in fname_list:
    #     floc = local_location+'/'+fname
    #     if os.path.isfile(floc) == False:
    #         # check for local download of files; if not, download them
    #         # choose each individual file name from the list
    #         download_ftp_files(remote_location, local_location, fname_list)

    # check for local download of files; if not, download them
    # choose each individual file name from the list
    download_ftp_files(remote_location, local_location, fname_list)

    # Combine all of the needed files into one dataframe
    full_kp_data = read_txt_files(fname_list, local_location)

    # Select the data we actually want
    time, kp = dm.slice_data_by_time(full_kp_data, ti, te)

    # When you are given certain times and want the associated Kp value, give the times in the expand variable,
    # And expand_kp will give back a list of Kp values at the given times
    if expand[0] != None:
        kp = dm.expand_kp(time, kp, expand)
        time = expand

    kp = kp.astype("float64")

    # I have the option to put in UT here. Not going to rn but could at a later point
    # Create an empty dataset at the time values that we made above
    kp_data = xr.Dataset(coords={"time": time})

    # Put the kp data into the dataset
    kp_data["Kp"] = xr.DataArray(kp, dims=["time"], coords={"time": time})

    return kp_data


def predict_efield_and_potential(
    model, time=None, data=None, return_pred=True, values_to_use=["Kp"]
):
    # A function that will take a model created by create_neural_network.py, and either a time or data argument,
    # and calculate the electric field and electric potential, plot them (if the user wants), and return the predicted values (if the user wants)

    # DOES NOT HAVE SYM-H FUNCTIONALITY

    # can input either the data you have that corresponds to the time you want to predict (aka the 5 hours of data, with the last input being the time you want to predict)
    # or you can input the time and the data will be downloaded for you
    # if both are given, the data will have priority
    if data is not None:
        complete_data = data
    elif time is not None:
        # we need the data from the 5 hours before the time to the time given
        # But the binned argument requires 1 day of data. so I do this instead
        # (!!) [BI:05/26/24] check time formatting
        ti = time - dt.timedelta(hours=5)
        te = time + dt.timedelta(minutes=5)

        mec_data = get_mec_data("mms1", "srvy", "l2", ti, te, binned=True)
        kp_data = get_kp_data_old(ti, te, expand=mec_data["time"].values)

        # (!!) [BI:05/26/24] bring back eventually with better loop, but we dont need these right now
        # dst_data = dd.get_dst_data_old(ti, te, expand=mec_data['time'].values)
        # symh_data = dd.get_symh_data_old(ti, te, expand=mec_data['time'].values)

        complete_data = xr.merge([mec_data, kp_data])
        # complete_data = xr.merge([mec_data, kp_data])
    elif time is None and data is None:
        raise TypeError("Either the desired time or the appropriate data must be given")
    ## !
    test_inputs = get_NN_inputs(
        complete_data, remove_nan=False, get_target_data=False, use_values=values_to_use
    )

    base_kp_values = test_inputs[-1].clone()

    number_of_inputs = len(values_to_use)
    size_of_input_vector = 60 * number_of_inputs + 3

    for L in range(4, 11):
        for MLT in range(0, 24):
            new_row = base_kp_values.clone()
            new_row[-3] = L
            new_row[-2] = np.cos(np.pi / 12 * MLT)
            new_row[-1] = np.sin(np.pi / 12 * MLT)
            even_newer_row = torch.empty((1, size_of_input_vector))
            even_newer_row[0] = new_row
            if L == 4 and MLT == 0:
                all_locations = even_newer_row
            else:
                all_locations = torch.cat((all_locations, even_newer_row))

    model.eval()
    with torch.no_grad():
        pred = model(all_locations)

    nL = 7
    nMLT = 24

    # Create a coordinate grid
    something = np.arange(0, 24)
    another_thing = np.concatenate(
        (something, something, something, something, something, something, something)
    ).reshape(nL, nMLT)
    phi = 2 * np.pi * another_thing / 24
    r = np.repeat(np.arange(4, 11), 24).reshape(nL, nMLT)

    # Start calculating the potential
    L = xr.DataArray(r, dims=["iL", "iMLT"])
    MLT = xr.DataArray(another_thing, dims=["iL", "iMLT"])

    # create an empty dataset and insert the predicted cartesian values into it. the time coordinate is nonsensical, but it needs to be time so that rot2polar works
    imef_data = xr.Dataset(
        coords={"L": L, "MLT": MLT, "polar": ["r", "phi"], "cartesian": ["x", "y", "z"]}
    )
    testing_something = xr.DataArray(
        pred,
        dims=["time", "cartesian"],
        coords={"time": np.arange(0, 168), "cartesian": ["x", "y", "z"]},
    )

    pred = pred.reshape(nL, nMLT, 3)

    # Create another dataset containing the locations around the earth as variables instead of dimensions
    imef_data["predicted_efield"] = xr.DataArray(
        pred, dims=["iL", "iMLT", "cartesian"], coords={"L": L, "MLT": MLT}
    )
    imef_data["R_sc"] = xr.DataArray(
        np.stack((r, phi), axis=-1).reshape(nL * nMLT, 2),
        dims=["time", "polar"],
        coords={"time": np.arange(0, 168), "polar": ["r", "phi"]},
    )

    pred.reshape(nL * nMLT, 3)

    # have to make sure that this actually works correctly. cause otherwise imma be getting some bad stuff
    # Convert the predicted cartesian values to polar
    imef_data["predicted_efield_polar"] = dm.rot2polar(
        testing_something, imef_data["R_sc"], "cartesian"
    ).assign_coords({"polar": ["r", "phi"]})

    # reshape the predicted polar data to be in terms of L and MLT, and put them into the same dataset
    somethingboi = imef_data["predicted_efield_polar"].values.reshape(nL, nMLT, 2)
    imef_data["predicted_efield_polar_iLiMLT"] = xr.DataArray(
        somethingboi, dims=["iL", "iMLT", "polar"], coords={"L": L, "MLT": MLT}
    )

    # [BI:5/28/24] for now, focus on E-Field...
    # potential = dm.calculate_potential(imef_data, 'predicted_efield_polar_iLiMLT')

    if return_pred == True:
        return imef_data  # , potential
