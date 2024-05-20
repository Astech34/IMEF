import numpy as np
import Neural_Networks as NN
import torch
import pandas as pd
# from torch.utils.data import DataLoader, TensorDataset

def random_undersampling(data, threshold=-40, quiet_storm_ratio=1.0):
    # Tbh not quite sure if I understood the algorithm correctly. It does what I think it wants, so I'll stick with it for now.
    # To be clear, this definitely undersamples the data. However the way I do it is by reducing the number of datapoints in each bin by a certain percentage. The authors may not do the same
    # But here is the link in case it's needed: https://link.springer.com/article/10.1007/s41060-017-0044-3

    intervals_of_storm_data = get_storm_intervals(data, threshold=threshold, max_hours=0)
    storm_counts=0
    for start, end in intervals_of_storm_data:
        storm_counts += end-start
    if storm_counts == 0:
        print(Warning('There is no storm data in the given dataset. Returning given data.'))
        return data
    quiet_counts=len(data['time'])-storm_counts

    if quiet_counts > storm_counts:
        bins_to_undersample = reverse_bins(intervals_of_storm_data, len(data['time']))
        percent_to_reduce = quiet_storm_ratio * storm_counts / quiet_counts

        if percent_to_reduce >= 1:
            raise ValueError('quiet_storm_ratio is too large. The max value for this dataset is '+str(quiet_counts/storm_counts))
        elif percent_to_reduce <= 0:
            raise ValueError('quiet_storm ratio is too small. It must be greater than 0')

        all_times = []
        for start, end in intervals_of_storm_data:
            all_times.append(data['time'].values[start:end])
        for start, end in bins_to_undersample:
            new_times_in_bin = np.random.choice(data['time'][start:end], int(percent_to_reduce * (end - start)), replace=False)
            all_times.append(new_times_in_bin)
        all_times = np.concatenate(all_times)
        all_times = np.sort(all_times)

        undersampled_data = data.sel(time=all_times)

        return undersampled_data
    else:
        # I don't know if a) this will ever come up, or b) if this did come up, we would want to undersample the storm data. So raising error for now
        print(Warning('There is more storm data than quiet data. Skipping undersampling.'))
        return data

def get_NN_inputs(imef_data, remove_nan=True, get_target_data=True, use_values=['Kp'], usetorch=True, undersample=None):
    # This could be made way more efficient if I were to make the function not store all the data even if it isn't used. But for sake of understandability (which this has little of anyways)
    # the random_undersampling argument should be a float, if it is not none. The float represents the quiet_storm_ratio

    if 'Kp' in use_values:
        Kp_data = imef_data['Kp']
    if 'Dst' in use_values:
        Dst_data = imef_data['Dst']
    if 'Symh' in use_values:
        Symh_data = imef_data['Sym-H']

    if remove_nan == True:
        imef_data = imef_data.where(np.isnan(imef_data['E_con'][:, 0]) == False, drop=True)
    if undersample != None and len(imef_data['time'].values) != 0:
        imef_data = random_undersampling(imef_data, quiet_storm_ratio=undersample)

    # Note that the first 5 hours of data cannot be used, since we do not have the preceding 5 hours of index data to get all the required data. Remove those 5 hours
    # Since there is now a try in the for loop, this shouldn't be needed. But just in case something strange comes up ill leave it here
    # imef_data = imef_data.where(imef_data['time']>=(imef_data['time'].values[0]+np.timedelta64(5, 'h')), drop=True)

    design_matrix_array=None
    if get_target_data == True:
        times_to_keep = []
    for counter in range(0, len(imef_data['time'].values)):
        new_data_line = []
        time_intervals = pd.date_range(end=imef_data['time'].values[counter], freq='5T', periods=60)
        try:
            if 'Kp' in use_values:
                Kp_index_data = Kp_data.sel(time=time_intervals).values.tolist()
                new_data_line += Kp_index_data
            if 'Dst' in use_values:
                Dst_index_data = Dst_data.sel(time=time_intervals).values.tolist()
                new_data_line += Dst_index_data
            if 'Symh' in use_values:
                Symh_index_data = Symh_data.sel(time=time_intervals).values.tolist()
                new_data_line += Symh_index_data

            # Along with the indices, we include 3 extra values to train on: The distance from the Earth (L), cos(MLT), and sin(MLT)
            # the_rest_of_the_data = np.array([imef_data['L'].values[counter], np.cos(np.pi / 12 * imef_data['MLT'].values[counter]), np.sin(np.pi / 12 * imef_data['MLT'].values[counter])]).tolist()

            # Along with the indices, we include 5 extra values to train on: The distance from the Earth (L), cos(MLT), sin(MLT), cos(MLAT), and sin(MLAT)
            the_rest_of_the_data = np.array([imef_data['L'].values[counter], np.cos(np.pi / 12 * imef_data['MLT'].values[counter]), np.sin(np.pi / 12 * imef_data['MLT'].values[counter]),
                                             np.cos(imef_data['MLAT'].values[counter]),np.sin(imef_data['MLAT'].values[counter])]).tolist()
            new_data_line += the_rest_of_the_data

            if design_matrix_array==None:
                design_matrix_array = [new_data_line]
            else:
                design_matrix_array.append(new_data_line)

            if get_target_data==True:
                times_to_keep.append(imef_data['time'].values[counter])
        except Exception as ex:
            # This should only be entered when there is not enough data to fully create the NN inputs required (aka previous 5 hours of index data, and location data)
            # print(ex)
            # raise ex
            pass

    if usetorch==True:
        design_matrix_array = torch.tensor(design_matrix_array)
    else:
        design_matrix_array = np.array(design_matrix_array)

    if get_target_data == True:
        efield_data = imef_data['E_con'].sel(time=times_to_keep).values

        if usetorch==True:
            efield_data = torch.from_numpy(efield_data)

        return design_matrix_array, efield_data
    else:
        return design_matrix_array