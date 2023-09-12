import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def slice_data(subset, position, times, spike_times, onset_column, offset_column):
    sliced_spikes = [[] for i in spike_times]
    for i in range(len(subset)):
        row = subset.iloc[i]

        # On- and offset definition within the trial.
        onset = row[onset_column].astype(int)
        offset = row[offset_column]

        pos_segment = position[onset:offset]
        time_segment = times[onset:offset]
        # print(f"{i}: position: {np.min(pos_segment)}-{np.max(pos_segment)}")
        # print(f'times: {np.min(time_segment)}-{np.max(time_segment)}')

        # Normalize
        pos_segment = (pos_segment - np.min(pos_segment)) / (
            np.max(pos_segment) - np.min(pos_segment)
        )

        for j, cell_spikes in enumerate(spike_times):
            sl_sp = [
                s for s in cell_spikes if (s >= times[onset] and s <= times[offset])
            ]
            # if len(sl_sp)>0:
            # print(f'min_spike = {np.min(sl_sp)}, max_spike = {np.max(sl_sp)}')
            sliced_spikes[j] += sl_sp

        if i == 0:
            norm_pos = pos_segment
            sliced_time = time_segment
        else:
            norm_pos = np.hstack([norm_pos, pos_segment])
            sliced_time = np.hstack([sliced_time, time_segment])

    return norm_pos, sliced_time, sliced_spikes


def skaggs_info_perspike(rate_map, occupancy_prob, epsilon=pow(10, -15)):
    """
    Calculates the Skaggs' information per spike for a single neuron.

    Parameters:
        rate_map (ndarray): A 1-dimensional array representing the rate map of the neuron.
            Each element corresponds to the firing rate at a specific location or bin.
        occupancy_prob (ndarray): A 1-dimensional array representing the occupancy histogram.
            Each element represents the probability of occupancy for a specific location or bin.
        epsilon (float, optional): A small value added to the rate to avoid division by zero.
            Defaults to 1e-15.

    Returns:
        float: The Skaggs' information per spike value for the neuron.
            It quantifies the spatial information content carried by the neuron's spikes.

    Notes:
        The Skaggs' information per spike is calculated using the following formula:
        sum(rate_map * np.log2((rate_map + epsilon) / avg_rate) * occupancy_prob) / avg_rate

        where avg_rate is the average firing rate of the neuron computed as:
        avg_rate = np.sum(rate_map * occupancy_prob)

        If the average firing rate (avg_rate) is less than or equal to epsilon, the function
        returns np.nan (not a number) to indicate that the result is undefined.

    Example:
        rate_map = np.array([1.5, 2.0, 0.8, 1.2])
        occupancy_prob = np.array([0.25, 0.35, 0.15, 0.25])
        info_per_spike = skaggs_info_perspike(rate_map, occupancy_prob)
        print(info_per_spike)  # Output: 0.851251...

    """
    rate_map = rate_map.flatten()
    occupancy_prob = occupancy_prob.flatten()
    avg_rate = np.sum(rate_map * occupancy_prob)
    if avg_rate > epsilon:
        return (
            sum(rate_map * np.log2((rate_map + epsilon) / avg_rate) * occupancy_prob)
            / avg_rate
        )
    else:
        return np.nan


def compute_firing_rate_maps(spike_positions, norm_pos):
    """
    Compute firing rate maps based on spike positions and normalized positions.

    Args:
        spike_positions (array-like): Array of spike positions.
        norm_pos (array-like): Array of normalized positions.

    Returns:
        numpy.ndarray: Firing rate maps.
        numpy.ndarray: Occupancy.
    """
    # Range of position = norm_pos, number of bins is 40 (1/0,025)
    space_bins = np.arange(0.0, 1.0, 0.025)
    vr_dt = 1 / 1000.0  # Frequency of VR-acquisition system

    # Compute histograms for each cell.
    spikes_hist = [np.histogram(s, space_bins)[0] for s in spike_positions]

    # Put them together into a matrix of floating-point numbers (for plotting).
    spikes_hist = np.vstack(spikes_hist).astype(np.float64)

    # Compute occupancy histogram to normalize the firing rate maps.
    occupancy = np.histogram(norm_pos, space_bins)[0] * vr_dt

    # Compute firing rate maps.
    firing_rate_maps = spikes_hist / occupancy

    return firing_rate_maps, occupancy


def calculate_spatial_info(firing_rate_maps, occupancy):
    """
    Calculate the spatial information for each neuron given firing rate maps and occupancy.

    Parameters:
        firing_rate_maps (numpy.ndarray): 2D array of firing rate maps, where each row represents a neuron and each column represents a spatial bin.
        occupancy (numpy.ndarray): 1D array of occupancy values for each spatial bin.

    Returns:
        spatial_info (numpy.ndarray): 1D array of spatial information values for each neuron.

    """
    # Calculate occupancy probability
    occupancy_prob = occupancy / np.sum(occupancy)

    # Calculate spatial information for each neuron
    spatial_info = []
    for neuron in firing_rate_maps:
        spatial_info.append(skaggs_info_perspike(neuron, occupancy_prob))

    # Convert spatial_info to NumPy array
    spatial_info = np.array(spatial_info)

    # Remove NaN values from spatial_info
    spatial_info = spatial_info[~np.isnan(spatial_info)]

    return spatial_info


def shuffle_spikes(spike_times, end_time, min_shift=10):
    shuffled_spikes = []
    for cell_spikes in spike_times:
        random_shift = np.random.uniform(min_shift, end_time)
        shifted_spikes = [(s + random_shift) % end_time for s in cell_spikes]
        shuffled_spikes.append(shifted_spikes)

    return shuffled_spikes


def process_sessions(
    animal_folder, animal_name, session_dates, results_df, save_plots, output_folder
):
    """
    Process behavioral and neural data from multiple sessions for a specific animal in different environments.

    Parameters:
        animal_folder (pathlib.Path): The folder containing the data for the animal.
        animal_name (str): The name of the animal.
        session_dates (list): List of strings representing the session dates.
        results_df (pd.DataFrame): DataFrame to store the processed results.
        save_plots (bool, optional): If True, plots will be saved for each session and environment. Default is False.

    Returns:
        pd.DataFrame: The updated results DataFrame after processing all the sessions.

    Note:
        This function assumes that the required processing functions are defined in a separate module named 'your_processing_module.py',
        and it imports these functions as 'pc'. Make sure to replace 'your_processing_module' with the actual name of your module.
        The function iterates over each session date, reads relevant data files (csv and pickle files), processes the data,
        computes firing rate maps, calculates spatial information, performs null hypothesis testing for place cell detection,
        saves plots (if required), and updates the results DataFrame with the processed information for each session and environment.
    """
    # Loop over session dates
    for date in session_dates:
        # Check if the current item is a .zip file
        if date.endswith(".zip"):
            continue

        data_path = animal_folder.joinpath(date)

        trial_data_file = data_path.joinpath("trial_data.csv")
        trial_data = pd.read_csv(trial_data_file)

        bdata_file = data_path.joinpath("behaviour_data.pickle")
        with open(bdata_file, "rb") as file:
            b_data = pickle.load(file)

        ndata_file = data_path.joinpath("neural_data.pickle")
        with open(ndata_file, "rb") as file:
            n_data = pickle.load(file)

        # Rest of your code for loading data
        scanner_fps = 30.0
        vr_fps = 1000.0

        end_time = n_data["traces"].shape[1] / scanner_fps
        scanner_times = np.arange(0, end_time, 1.0 / scanner_fps)

        position = b_data["position"]
        times = b_data["time"]
        spikes = n_data["deconvolved"]

        spike_times = []
        for s in spikes:
            spike_times.append(scanner_times[s])

        # Perform operations for different env settings
        for env in [1, 2, 3]:
            subset = trial_data[trial_data["env_label"] == env]
            onset_column = "env_onset"
            offset_column = "tunnel1_onset"
            norm_pos, sliced_time, sliced_spikes = slice_data(
                subset, position, times, spike_times, onset_column, offset_column
            )
            spike_positions = [
                np.interp(s, sliced_time, norm_pos) for s in sliced_spikes
            ]

            # Rest of your code for processing the current session and env setting
            # Perform additional operations
            firing_rate_maps, occupancy = compute_firing_rate_maps(
                spike_positions, norm_pos
            )
            spatial_info = calculate_spatial_info(firing_rate_maps, occupancy)

            # NULL
            n_shuff = 20
            null_spatial_info_distr = []

            for n in range(n_shuff):
                shuff_spike_times = shuffle_spikes(spike_times, scanner_times[-1])
                shuff_norm_pos, shuff_sliced_time, shuff_sliced_spikes = slice_data(
                    subset,
                    position,
                    times,
                    shuff_spike_times,
                    onset_column,
                    offset_column,
                )
                shuff_spike_positions = [
                    np.interp(s, shuff_sliced_time, shuff_norm_pos)
                    for s in shuff_sliced_spikes
                ]

                shuff_firing_rate_maps, shuff_occupancy = compute_firing_rate_maps(
                    shuff_spike_positions, shuff_norm_pos
                )
                shuff_spatial_info = calculate_spatial_info(
                    shuff_firing_rate_maps, shuff_occupancy
                )

                for s in shuff_spatial_info:
                    null_spatial_info_distr.append(s)

            # Save plots if required
            if save_plots:
                session_output_folder = output_folder.joinpath(f"{animal_name}_{date}")
                session_output_folder.mkdir(exist_ok=True, parents=True)

                # Plotting per env
                # norm_maps = firing_rate_maps / np.max(firing_rate_maps, axis=1)[:, np.newaxis]
                norm_maps = firing_rate_maps / (
                    np.max(firing_rate_maps, axis=1)[:, np.newaxis] + 1e-8
                )  # small epsilon number to avoid /0 error

                plt.figure(figsize=(15, 5))

                # Find the location of the peak firing rate for each cell
                peak_locations = norm_maps.argmax(axis=1)

                # Sort the firing rate maps based on the peak locations
                ix = np.argsort(peak_locations)

                # Plot the sorted firing rate maps
                plt.imshow(norm_maps[ix, :], cmap="inferno", aspect="auto")

                # Set the x-axis label
                plt.xlabel("location (bins)")

                # Set the y-axis label
                plt.ylabel("cell #")

                # Add a colorbar to the plot
                plt.colorbar()

                # Save the ratemaps plot
                ratemaps_folder = session_output_folder.joinpath("ratemaps")
                ratemaps_folder.mkdir(exist_ok=True)
                plt.savefig(
                    ratemaps_folder.joinpath(
                        f"{animal_name}_{date}_env_{env}_firing_rate_maps.png"
                    )
                )
                plt.close()  # Close figure to save memory

                # Plotting null_spatial_info_distr
                plt.figure(figsize=(5, 5))
                plt.hist(null_spatial_info_distr, density=True, bins=20, label="Null")
                plt.hist(
                    spatial_info, density=True, alpha=0.5, bins=20, label="Experimental"
                )
                sns.despine()
                plt.xlabel("spatial info (bit/spike)")
                plt.ylabel("probability density")
                plt.legend()

                # Save the spatial_distr plot
                spatial_info_folder = session_output_folder.joinpath(
                    "spatial_info_distr"
                )
                spatial_info_folder.mkdir(exist_ok=True)
                plt.savefig(
                    spatial_info_folder.joinpath(
                        f"{animal_name}_{date}_env_{env}_null_spatial_info_distr_plot.png"
                    )
                )
                plt.close()

            # Calculate place cell statistics
            place_cell_th = np.percentile(null_spatial_info_distr, 95)
            n_place_cells = sum(i > place_cell_th for i in spatial_info)
            fraction = n_place_cells / len(spatial_info)

            # Create a DataFrame with the current session's results
            session_results = pd.DataFrame(
                {
                    "Animal": [animal_name],
                    "Session Date": [date],
                    "Environment": [env],
                    "Number of Place Cells": [n_place_cells],
                    "Fraction": [fraction],
                }
            )

            # Concatenate the session results with the main results_df
            results_df = pd.concat([results_df, session_results], ignore_index=True)

    # Return the updated results dataframe
    return results_df


def process_sessions_rz(
    animal_folder, animal_name, session_dates, results_df, save_plots, output_folder
):
    """
    Process behavioral and neural data from multiple sessions for a specific animal in different reward zones (environments).

    Parameters:
        animal_folder (pathlib.Path): The folder containing the data for the animal.
        animal_name (str): The name of the animal.
        session_dates (list): List of strings representing the session dates.
        results_df (pd.DataFrame): DataFrame to store the processed results.
        save_plots (bool, optional): If True, plots will be saved for each session and reward zone environment. Default is False.

    Returns:
        pd.DataFrame: The updated results DataFrame after processing all the sessions.

    Note:
        This function assumes that the required processing functions are defined in a separate module named 'your_processing_module.py',
        and it imports these functions as 'pc'. Make sure to replace 'your_processing_module' with the actual name of your module.
        The function iterates over each session date, reads relevant data files (csv and pickle files), processes the data,
        computes firing rate maps, calculates spatial information, performs null hypothesis testing for place cell detection,
        saves plots (if required), and updates the results DataFrame with the processed information for each session and reward zone environment.
    """
    # Loop over session dates
    for date in session_dates:
        # Check if the current item is a .zip file
        if date.endswith(".zip"):
            continue

        data_path = animal_folder.joinpath(date)

        trial_data_file = data_path.joinpath("trial_data.csv")
        trial_data = pd.read_csv(trial_data_file)

        bdata_file = data_path.joinpath("behaviour_data.pickle")
        with open(bdata_file, "rb") as file:
            b_data = pickle.load(file)

        ndata_file = data_path.joinpath("neural_data.pickle")
        with open(ndata_file, "rb") as file:
            n_data = pickle.load(file)

        # Rest of your code for loading data
        scanner_fps = 30.0
        vr_fps = 1000.0

        end_time = n_data["traces"].shape[1] / scanner_fps
        scanner_times = np.arange(0, end_time, 1.0 / scanner_fps)

        position = b_data["position"]
        times = b_data["time"]
        spikes = n_data["deconvolved"]

        spike_times = []
        for s in spikes:
            spike_times.append(scanner_times[s])

        # Perform operations for different env settings
        for env in [1, 2, 3]:
            subset = trial_data[trial_data["env_label"] == env]
            onset_column = "reward_zone_onset"
            offset_column = "tunnel2_onset"
            norm_pos, sliced_time, sliced_spikes = slice_data(
                subset, position, times, spike_times, onset_column, offset_column
            )
            spike_positions = [
                np.interp(s, sliced_time, norm_pos) for s in sliced_spikes
            ]

            # Rest of your code for processing the current session and env setting
            # Perform additional operations
            firing_rate_maps, occupancy = compute_firing_rate_maps(
                spike_positions, norm_pos
            )
            spatial_info = calculate_spatial_info(firing_rate_maps, occupancy)

            # NULL
            n_shuff = 20
            null_spatial_info_distr = []

            for n in range(n_shuff):
                shuff_spike_times = shuffle_spikes(spike_times, scanner_times[-1])
                shuff_norm_pos, shuff_sliced_time, shuff_sliced_spikes = slice_data(
                    subset,
                    position,
                    times,
                    shuff_spike_times,
                    onset_column,
                    offset_column,
                )
                shuff_spike_positions = [
                    np.interp(s, shuff_sliced_time, shuff_norm_pos)
                    for s in shuff_sliced_spikes
                ]

                shuff_firing_rate_maps, shuff_occupancy = compute_firing_rate_maps(
                    shuff_spike_positions, shuff_norm_pos
                )
                shuff_spatial_info = calculate_spatial_info(
                    shuff_firing_rate_maps, shuff_occupancy
                )

                for s in shuff_spatial_info:
                    null_spatial_info_distr.append(s)

            # Save plots if required
            if save_plots:
                session_output_folder = output_folder.joinpath(f"{animal_name}_{date}")
                session_output_folder.mkdir(exist_ok=True, parents=True)

                # Plotting per env
                # norm_maps = firing_rate_maps / np.max(firing_rate_maps, axis=1)[:, np.newaxis]
                norm_maps = firing_rate_maps / (
                    np.max(firing_rate_maps, axis=1)[:, np.newaxis] + 1e-8
                )  # small epsilon number to avoid /0 error

                plt.figure(figsize=(15, 5))

                # Find the location of the peak firing rate for each cell
                peak_locations = norm_maps.argmax(axis=1)

                # Sort the firing rate maps based on the peak locations
                ix = np.argsort(peak_locations)

                # Plot the sorted firing rate maps
                plt.imshow(norm_maps[ix, :], cmap="inferno", aspect="auto")

                # Set the x-axis label
                plt.xlabel("location (bins)")

                # Set the y-axis label
                plt.ylabel("cell #")

                # Add a colorbar to the plot
                plt.colorbar()

                # Save the ratemaps plot
                ratemaps_folder = session_output_folder.joinpath("ratemaps_rz")
                ratemaps_folder.mkdir(exist_ok=True)
                plt.savefig(
                    ratemaps_folder.joinpath(
                        f"{animal_name}_{date}_env_{env}_firing_rate_maps.png"
                    )
                )
                plt.close()  # Close figure to save memory

                # Plotting null_spatial_info_distr
                plt.figure(figsize=(5, 5))
                plt.hist(null_spatial_info_distr, density=True, bins=20, label="Null")
                plt.hist(
                    spatial_info, density=True, alpha=0.5, bins=20, label="Experimental"
                )
                sns.despine()
                plt.xlabel("spatial info (bit/spike)")
                plt.ylabel("probability density")
                plt.legend()

                # Save the spatial_distr plot
                spatial_info_folder = session_output_folder.joinpath(
                    "spatial_info_distr_rz"
                )
                spatial_info_folder.mkdir(exist_ok=True)
                plt.savefig(
                    spatial_info_folder.joinpath(
                        f"{animal_name}_{date}_env_{env}_null_spatial_info_distr_plot.png"
                    )
                )
                plt.close()

            # Calculate place cell statistics
            place_cell_th = np.percentile(null_spatial_info_distr, 95)
            n_place_cells = sum(i > place_cell_th for i in spatial_info)
            fraction = n_place_cells / len(spatial_info)

            # Create a DataFrame with the current session's results
            session_results = pd.DataFrame(
                {
                    "Animal": [animal_name],
                    "Session Date": [date],
                    "Environment": [env],
                    "Number of Place Cells": [n_place_cells],
                    "Fraction": [fraction],
                }
            )

            # Concatenate the session results with the main results_df
            results_df = pd.concat([results_df, session_results], ignore_index=True)

    # Return the updated results dataframe
    return results_df


def calculate_place_cell_statistics(null_spatial_info_distr, spatial_info):
    """
    Calculate place cell statistics.

    Parameters:
    - null_spatial_info_distr (numpy.ndarray): An array containing spatial information values from null distribution.
    - spatial_info (numpy.ndarray): An array containing spatial information values.

    Returns:
    - place_cell_th (float): The 95th percentile of the null_spatial_info_distr.
    - n_place_cells (int): The number of place cells (values in spatial_info greater than place_cell_th).
    - fraction (float): The fraction of place cells in spatial_info.
    """
    place_cell_th = np.percentile(null_spatial_info_distr, 95)
    n_place_cells = sum(i > place_cell_th for i in spatial_info)
    fraction = n_place_cells / len(spatial_info)

    return place_cell_th, n_place_cells, fraction
