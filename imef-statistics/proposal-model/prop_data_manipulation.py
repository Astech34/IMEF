import numpy as np
import xarray as xr
import datetime as dt
from datetime import timezone
from warnings import warn
import scipy.optimize as optimize
from scipy.stats import binned_statistic, binned_statistic_2d, binned_statistic_dd
import torch
import pandas as pd
import os

# For debugging purposes
# np.set_printoptions(threshold=np.inf)

R_E = 6378.1  # km

def create_timestamps(data, ti, te):
    '''
    Convert times to UNIX timestamps.
    Parameters
    ----------
    data : `xarray.DataArray`
        Data with coordinate 'time' containing the time stamps (np.datetime64)
    ti, te : `datetime.datetime`
        Time interval of the data
    Returns
    -------
    ti, te : float
        Time interval converted to UNIX timestamps
    timestamps :
        Time stamps converted to UNIX times
    '''

    # [BI:5/27/24] added some clarifying documentation, no major changes. May want to adress some stuff in future.

    # define the epoch and one second in np.datetime 6 so one can convert np.datetime64 objects to timestamp values
    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')

    # round up the end time by one microsecond so the bins aren't marginally incorrect.
    #   - this is good enough for now, but this is a line to keep an eye on, as it will cause some strange things to 
    #   happen due to the daylight savings fix later on.
    #   - (!) if off by 1 microsecond, will erroneously gain/lose 1 hour
    if (te.second != 0) and te.second != 30:
        te = te + dt.timedelta(microseconds=1)

    # set initial and end times
    ti_datetime = ti
    te_datetime = te

    # convert the start and end times to a unix timestamp
    #   - this section adapts for the 4 hour time difference (in seconds) that timestamp() automatically applies to datetime. 
    #   Otherwise the times from data and these time will not line up right
    #   - this appears because timestamp() corrects for local time difference, while the np.datetime64 method did not
    #       - this could be reversed and added to all the times in data, but I chose this way.
    #   - (*) note that this can cause shifts in hours if the timezone changes
    ti = ti.timestamp() - 14400
    te = te.timestamp() - 14400

    # this is to account for daylight savings
    # [BI:5/27/24] probably fine for now, want to look at in the future...
    #   - lazy fix: check to see if ti-te is the right amount of time. If yes, move on. If no, fix by changing te to what it should be
    #   - this forces the input time to be specifically 1 day of data, otherwise this number doesn't work.
    #   - though maybe the 86400 could be generalized using te-ti before converting to timestamp. Possible change there
    #   - though UTC is definitely something to be done, time permitting (i guess it is in UTC. need to figure out at some point)
    #   - this will only work for exactly 1 day of data being downloaded. It will be fine for sample and store_edi data,
    #   - however if running a big download that goes through a daylight savings day, there will be an issue

    if ti_datetime + dt.timedelta(days=1) == te_datetime:
        if te - ti > 86400:
            te -= 3600
        elif te - ti < 86400:
            te += 3600

    # create the array where the unix timestamp values go
    #   - the timestamp values are needed so we can bin the values with binned_statistic
    #   - (*) note that the data argument must be an xarray object with a 'time' dimension so that this works. Could be generalized at some point
    timestamps = (data['time'].values - unix_epoch) / one_second

    # return timestamps
    return ti, te, timestamps


def get_5min_times(data, vars_to_bin, timestamps, ti, te):

    # [BI:5/27/24] added some clarifying documentation, no major changes.
    # get the times here. This way we don't have to rerun getting the times for every single variable that is being binned

    # [BI:5/27/24] why 300 Izzak? maybe make var nbins=300...
    number_of_bins = (te - ti) / 300

    # compute a binned statistic for one or more sets of data (~ generalized histogram function)
    # bins data to timestamp range
    # returns:
    #   - array; the values of the selected statistic in each bin.
    #   - bin_edges (float) return the bin edges (length(statistic)+1).
    #   - binnumber(1D int array) indices of the bins (corresponding to bin_edges) in which each value of x belongs.

    count, bin_edges, binnum = binned_statistic(x=timestamps, values=data[vars_to_bin[0]], statistic='count',
                                                bins=number_of_bins, range=(ti, te))

    # create an nparray where the new 5 minute interval datetime64 objects will go
    new_times = np.array([], dtype=object)

    # create the datetime objects and add them to new_times
    for time in bin_edges:
        # Don't run if the time value is the last index in bin_edges. There is 1 more bin edge than there is mean values
        # This is because bin_edges includes an extra edge to encompass all the means
        # As a result, the last bin edge (when shifted to be in the middle of the dataset) doesn't correspond to a mean value
        # So it must be ignored so that the data will fit into a new dataset
        if time != bin_edges[-1]:
            # convert timestamp to datetime object
            new_time = dt.datetime.fromtimestamp(time, tz=timezone.utc) # [BI] new version, should work...
            # new_time = dt.datetime.utcfromtimestamp(time) # [BI] old, function is deprectaed
            # if not working, see: https://blog.ganssle.io/articles/2019/11/utcnow.html

            # add 2.5 minutes to place the time in the middle of each bin, rather than the beginning
            new_time = new_time + dt.timedelta(minutes=2.5)

            # add the object to the nparray
            new_times = np.append(new_times, [new_time])

    # return the datetime objects of the 5 minute intervals created in binned_statistic
    return new_times


def bin_5min(data, vars_to_bin, index_names, ti, te):
    # [BI:5/28/24] added some clarifying documentation, no major changes, but this can DEFINITLEY be made more efficient
    '''
    Bin one day's worth of data into 5-minute intervals.
    Parameters
    ----------
    data
    vars_to_bin [BI:5/28/24] what do these mean???
    index_names [BI:5/28/24] list of all column names in array??
    ti, te
    Returns
    -------
    complete_data
    '''
    # Any variables that are not in var_to_bin are lost (As they can't be mapped to the new times otherwise)
    # Note that it is possible that NaN values appear in the final xarray object. This is because there were no data points in those bins
    # To remove these values, use xarray_object = test.where(np.isnan(test['variable_name']) == False, drop=True) (Variable has no indices)
    # Or xarray_object = xarray_object.where(np.isnan(test['variable_name'][:,0]) == False, drop=True) (With indices)

    # Also note that in order for this to work correctly, te-ti must be a multiple of 5 minutes.
    # This is addressed in the get_xxx_data functions, since they just extend the downloaded times by an extra couple minutes or whatever

    # In order to bin the values properly, we need to convert the datetime objects to integers. I chose to use unix timestamps to do so
    ti, te, timestamps = create_timestamps(data, ti, te)
    new_times = get_5min_times(data, vars_to_bin, timestamps, ti, te)

    # [BI:5/27/24] why 300 Izzak? maybe make var nbins=300... (see get_5min_times)
    number_of_bins = (te - ti) / 300

    # iterate through every variable (and associated index) in the given list
    for var_counter in range(len(vars_to_bin)):
        if index_names[var_counter] == '':
            # Since there is no index associated with this variable, there is only 1 thing to be meaned. So take the mean of the desired variable
            # [BI:5/28/24] might wanna look into this again...
            means, bin_edges_again, binnum = binned_statistic(x=timestamps, values=data[vars_to_bin[var_counter]],
                                                              statistic='mean', bins=number_of_bins, range=(ti, te))
            std, bin_edges_again, binnum = binned_statistic(x=timestamps, values=data[vars_to_bin[var_counter]],
                                                            statistic='std', bins=number_of_bins, range=(ti, te))

            # Create the dataset for the meaned variable
            new_data = xr.Dataset(coords={'time': new_times})

            # Fix the array so it will fit into the dataset
            var_values = means.T
            var_values_std = std.T

            # Put the data into the dataset
            new_data[vars_to_bin[var_counter]] = xr.DataArray(var_values, dims=['time'], coords={'time': new_times})
            new_data[vars_to_bin[var_counter] + '_std'] = xr.DataArray(var_values_std, dims=['time'],
                                                                       coords={'time': new_times})
        else:
            # Empty array where the mean of the desired variable will go
            means = np.array([[]])
            stds = np.array([[]])

            # Iterate through every variable in the associated index
            for counter in range(len(data[index_names[var_counter] + '_index'])):
                # Find the mean of var_to_bin
                # mean is the mean in each bin, bin_edges is the edges of each bin in timestamp values, and binnum is which values go in which bin
                mean, bin_edges_again, binnum = binned_statistic(x=timestamps,
                                                                 values=data[vars_to_bin[var_counter]][:, counter],
                                                                 statistic='mean', bins=number_of_bins, range=(ti, te))
                std, bin_edges_again, binnum = binned_statistic(x=timestamps,
                                                                values=data[vars_to_bin[var_counter]][:, counter],
                                                                statistic='std', bins=number_of_bins, range=(ti, te))

                # If there are no means yet, designate the solved mean value as the array where all of the means will be stored. Otherwise combine with existing data
                if means[0].size == 0:
                    means = [mean]
                    stds = [std]
                else:
                    means = np.append(means, [mean], axis=0)
                    stds = np.append(stds, [std], axis=0)

            # Create the new dataset where the 5 minute bins will go
            new_data = xr.Dataset(coords={'time': new_times, index_names[var_counter] + '_index': data[
                index_names[var_counter] + '_index']})

            # Format the mean values together so that they will fit into new_data
            var_values = means.T
            var_values_std = stds.T

            # Put in var_values
            new_data[vars_to_bin[var_counter]] = xr.DataArray(var_values,
                                                              dims=['time', index_names[var_counter] + '_index'],
                                                              coords={'time': new_times})
            new_data[vars_to_bin[var_counter] + '_std'] = xr.DataArray(var_values_std, dims=['time', index_names[
                var_counter] + '_index'], coords={'time': new_times})

        # If this is the first run, designate the created data as the dataset that will hold all the data. Otherwise combine with the existing data
        if var_counter == 0:
            complete_data = new_data
        else:
            complete_data = xr.merge([complete_data, new_data])

    return complete_data


def slice_data_by_time(full_data, ti, te):
    # [BI:6/06/24] added some clarifying documentation, no major changes, although I think this could be made more efficient.
    # [BI:5/28/24] what is wanted_value?
    # arrays for desired time and data values
    time = np.array([]) # array to hold time values
    wanted_value = np.array([])

    # slice the wanted data and put them into 2 lists
    for counter in range(0, len(full_data)):
        # slice data by single space
        #   - (the data at each index is all in one line, separated by whitespace)
        new = str.split(full_data.iloc[counter][0])

        # create time string at given point
        time_str = str(new[0]) + '-' + str(new[1]) + '-' + str(new[2]) + 'T' + str(new[3][:2]) + ':00:00'

        # create datetime obj
        insert_time_beg = dt.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')

        # adjust time to account for Kp binning
        #   - for Kp, middle of the bin is 1.5 hours past the beginning of the bin (which is insert_time_beg)
        insert_time_mid = insert_time_beg + dt.timedelta(hours=1.5)

        # if the data point is within the time range that is desired, insert the time and the associated kp index
        # the other datasets use the dates as datetime64 objects. So must use those instead of regular datetime objects
        if insert_time_mid + dt.timedelta(hours=1.5) > ti and insert_time_mid - dt.timedelta(hours=1.5) < te:
            insert_kp = new[7]
            time = np.append(time, [insert_time_mid])
            wanted_value = np.append(wanted_value, [insert_kp])

    return time, wanted_value


def expand_kp(kp_times, kp, time_to_expand_to):
    # [BI:6/06/24] added some clarifying documentation, no major changes.
    #   - this function is capable of doing the same thing as expand_5min, and more. [see Izzak's code, data_manipulation.py]
    #   - this function can be used for other indices and such as long as they are inputted in the same format as Kp data

    # datetime objects that are placed into xarrays get transformed into datetime64 objects, and the conventional methods of changing them back do not seem to work
    #   - therefore, You have to make a datetime64 version of the kp_times so that they can be subtracted correctly

    # iterate through all times and convert them to datetime64 objects
    # # [BI:6/06/24] if this becomes a common need, make separate function?
    for time in kp_times:
        # timedelta is done because the min function used later chooses the lower value in the case of a tie > want the upper value to be chosen
        if type(time) == type(dt.datetime(2015, 9, 10)):
            time64 = np.datetime64(time - dt.timedelta(microseconds=1))
        elif type(time) == type(np.datetime64(1, "Y")):
            time64 = time - np.timedelta64(
                1, "ms"
            )  # This thing only works for 1 millisecond, not 1 microsecond. Very sad # [BI:6/06/24] leaving this comment as is.
        else:
            raise TypeError(
                "Time array must contain either datetime or datetime64 objects"
            )

        if time == kp_times[0]:
            datetime64_kp_times = np.array([time64])
        else:
            datetime64_kp_times = np.append(datetime64_kp_times, [time64])

    # find the date closest to each time given
    #   - in other words, it is used to find the closest time to each time from the given list
    absolute_difference_function = lambda list_value: abs(list_value - given_value)

    # iterate through all times that to expand kp to
    for given_value in time_to_expand_to:

        # find closest value
        closest_value = min(datetime64_kp_times, key=absolute_difference_function)

        # find the corresponding index of the closest value
        index = np.where(datetime64_kp_times == closest_value)

        if given_value == time_to_expand_to[0]:
            # if this is the first iteration, create an ndarray containing the kp value at the corresponding index
            new_kp = np.array([kp[index]])
        else:
            # if this is not the first iteration, combine the new kp value with the existing ones
            new_kp = np.append(new_kp, kp[index])

    return new_kp


def rot2polar(vec, pos, dim):
    """
    Rotate vector from cartesian coordinates to polar coordinates
    """
    # Polar unit vectors
    phi = pos.loc[:, "phi"]
    r_hat = xr.concat(
        [np.cos(phi).expand_dims(dim), np.sin(phi).expand_dims(dim)], dim=dim
    )
    phi_hat = xr.concat(
        [-np.sin(phi).expand_dims(dim), np.cos(phi).expand_dims(dim)], dim=dim
    )

    # Rotate vector to polar coordinates
    Vr = vec[:, [0, 1]].dot(r_hat, dims=dim)
    Vphi = vec[:, [0, 1]].dot(phi_hat, dims=dim)
    v_polar = xr.concat([Vr, Vphi], dim="polar").T.assign_coords(
        {"polar": ["r", "phi"]}
    )

    return v_polar
