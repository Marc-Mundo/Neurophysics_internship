import pickle
import numpy as np
import matplotlib.pyplot as plt


def normpos_slicedtime(subset, b_data, onset_col, offset_col):
    """
    Extracts position and time information from behavioural data based on provided onset and offset columns.

    Args:
        subset (DataFrame): Subset dataframe containing trial information.
        b_data (dict): Behavioural data dictionary with 'position' and 'time' keys.
        onset_col (str): Column name for the onset values in the subset dataframe.
        offset_col (str): Column name for the offset values in the subset dataframe.

    Returns:
        tuple: A tuple containing the normalized position (norm_pos) and sliced time (sliced_time).

    """

    # Extract the position and time information out of the behavioural data.
    position = b_data["position"]
    times = b_data["time"]

    # Subset selection.
    for i in range(len(subset)):
        row = subset.iloc[i]

        # On- and offset definition within the trial.
        onset = row[onset_col].astype(int)
        offset = row[offset_col]

        pos_segment = position[onset:offset]
        time_segment = times[onset:offset]
        print(f"{i}: {np.max(pos_segment)-np.min(pos_segment)}")

        # Normalize
        pos_segment = (pos_segment - np.min(pos_segment)) / (
            np.max(pos_segment) - np.min(pos_segment)
        )

        if i == 0:
            norm_pos = pos_segment
            sliced_time = time_segment
        else:
            norm_pos = np.hstack([norm_pos, pos_segment])
            sliced_time = np.hstack([sliced_time, time_segment])

    return norm_pos, sliced_time


def get_spike_positions(n_data, norm_pos, b_data):
    """
    Compute the spike positions based on the given n_data, norm_pos, and b_data.

    Args:
        n_data (numpy.ndarray): Data containing spike information.
        norm_pos (numpy.ndarray): Normalized positions.
        b_data (numpy.ndarray): Data containing time information.

    Returns:
        numpy.ndarray: Array of spike positions.

    """
    spikes = n_data["deconvolved"]
    dt = 1 / 30.0  # scanner recording frequency
    end_time = n_data["traces"].shape[1]
    scanner_times = np.arange(0, end_time, dt)

    spike_times = []
    for s in spikes:
        spike_times.append(scanner_times[s])

    t = b_data["time"][: len(norm_pos)]

    spike_positions = [np.interp(s, t, norm_pos) for s in spike_times]

    return spike_positions


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

    space_bins = np.arange(0.0, 1.0, 0.05)  # Range of position = norm_pos
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
