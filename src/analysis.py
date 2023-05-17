import numpy as np


def lick_counter(trial_duration, lick_start, lick_end, licks):
    """
    Counts the number of licks that occur during a specified time window and calculates the fraction of licks that
    occur at the reward onset time.

    PARAMETERS:
        trial_duration (float): The duration of the trial in seconds.
        lick_start (int): The timepoint at which the reward was presented.
        lick_end (int): The timepoint at which the reward window ends.
        licks (numpy array): A numpy array of shape (m,) containing the timepoints at which licks occurred.

    RETURNS:
        tuple: A tuple containing the lick count and fraction of licks at the reward onset time.
    """

    # Initialize the lick counter.
    lick_counter = 0

    # Iterate through each row in the dataframe.
    for i in range(len(lick_start)):
        # Check if the lick time falls within the reward window.
        for lick in licks:
            if lick_start[i] <= lick <= lick_end[i]:
                lick_counter += 1

    # Compute the fraction of licks that occur at the reward onset time.
    reward_licks = lick_counter / len(licks) * 100

    return lick_counter, reward_licks


def compute_feature_position(timestamp, position, min_pos, max_pos):
    """
    Takes the time of an event (in index), the position array, the min of the trial position and the max
    of the trial position, returns the normalized position of the event.

    PARAMETERS:
    timestamp (int) : timestamp of the event (in index).
    position (numpy.ndarray) : position array.
    min_pos (float) : minimum position value of the trial.
    max_pos (float) : maximum position value of the trial.

    RETURNS:
    norm_pos (float): normalized position of the event.
    """

    # Extract the position value at the given timestamp.
    pos = position[int(timestamp)]

    # Calculate the normalized position of the event based on the min and max position values.
    norm_pos = (pos - min_pos) / (max_pos - min_pos)

    # Return the normalized position value.
    return norm_pos


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def position_event_histogram(b_data, trial_data):
    """
    Plot a histogram of the normalized position with markers for specific events:
    Reward zone onset, Tunnel1 onset, Sound onset

    PARAMETERS:
    b_data (pandas.DataFrame) : a DataFrame containing the behavior data.
    trial_data (pandas.DataFrame) : a DataFrame containing the trial data.

    RETURNS:
    A histogram plot of the normalized position data with markers for the event onsets.
    """

    # Get the position data from the behavior data DataFrame.
    position = b_data["position"]

    # Initialize lists to store the positions of the event onsets.
    rz_onsets = []
    tunnel1_onsets = []
    sound_onsets = []

    # Loop through each trial in the trial data DataFrame.
    for i in range(len(trial_data)):
        # Get the current trial row.
        row = trial_data.iloc[i]

        # Extract the onset and offset times for the current trial segment.
        onset = row["env_onset"].astype(int)
        offset = row["tunnel2_offset"]

        # Get the position data for the current trial segment.
        pos_segment = position[onset:offset]

        # Compute the minimum and maximum positions in the current trial segment.
        min_pos = np.min(pos_segment)
        max_pos = np.max(pos_segment)

        # Compute the positions of the reward zone onset, tunnel 1 onset, and sound onset.
        rz_pos = compute_feature_position(
            row["reward_zone_onset"], position, min_pos, max_pos
        )
        rz_onsets.append(rz_pos)

        t1_pos = compute_feature_position(
            row["tunnel1_onset"], position, min_pos, max_pos
        )
        tunnel1_onsets.append(t1_pos)

        if pd.notnull(row["sound_onset"]):  # Skip NaNs.
            sound_pos = compute_feature_position(
                int(row["sound_onset"]), position, min_pos, max_pos
            )
            sound_onsets.append(sound_pos)

        # Normalize the position data for the current trial segment.
        pos_segment = (pos_segment - np.min(pos_segment)) / (
            np.max(pos_segment) - np.min(pos_segment)
        )

        # Concatenate the normalized position data for the current trial segment with the previous segments.
        if i == 0:
            norm_pos = pos_segment
        else:
            norm_pos = np.hstack([norm_pos, pos_segment])

    # Plot a histogram of the normalized position data with markers for the event onsets.
    plt.hist(norm_pos, bins=30, density=True)
    plt.axvline(x=np.mean(rz_onsets), c="r")
    plt.axvline(x=np.mean(tunnel1_onsets), c="g")
    plt.axvline(x=np.nanmean(sound_onsets), c="m")
    plt.show()


import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


def compute_velocity(
    b_data, dt=1.0 / 1000.0, pos_sigma=2, vel_sigma=50, vel_win=10000, vel_smooth=100
):
    """
    Computes the velocity matrix from the position data in b_data using Gaussian filtering.

    PARAMETERS:
    b_data (dict): A dictionary containing the position data as a numpy array under the key 'position'.
    dt (float, optional): The time step between position measurements, in seconds. Default of recording apparatus is 1/1000 Hz.
    pos_sigma (float, optional): The standard deviation of the Gaussian kernel for smoothing the position data.
                                     Default is 2.
    vel_sigma (float, optional): The standard deviation of the Gaussian kernel for smoothing the velocity data.
                                     Default is 50.
    vel_win (int, optional): The size of the window for computing the velocity, in samples. Default is 10000.
    vel_smooth (int, optional): The size of the Gaussian smoothing kernel for the velocity data. Default is 100.

    RETURNS:
    tuple: A tuple containing the velocity matrix and a plot of the velocity data.
    """

    # Apply Gaussian smoothing to the position data to reduce noise.
    pos = gaussian_filter1d(b_data["position"].astype(float), sigma=pos_sigma)

    # Calculate the discrete difference between adjacent position samples to obtain the velocity.
    vel = np.diff(pos) / dt

    # Apply Gaussian smoothing to the velocity data to further reduce noise.
    vel = gaussian_filter1d(vel, sigma=vel_sigma)

    # Plot a window of the velocity data with additional Gaussian smoothing for visualization.
    plt.plot(gaussian_filter1d(vel[:vel_win], sigma=vel_smooth))

    # Add labels and title to the velocity plot.
    plt.xlabel("Frame number")
    plt.ylabel("Velocity (m/s)")
    # plt.title('Velocity plot')

    # Show the plot.
    plt.show()

    # Return the velocity data.
    return vel


import matplotlib.pyplot as plt
from scipy.stats import binned_statistic


def pos_vel_scatterplot(norm_pos, vel, nbins=50):
    """
    Create a scatter plot of the relationship between position and velocity
    using binned statistics.

    PARAMETERS:
    norm_pos : array-like
            The normalized positions.
    vel : array-like
            The velocities.
    nbins : int, optional
            The number of bins to use for the scatter plot (default=50).

    RETURNS:
    A scatterplot with graph between datapoints to visualize the relationship between position and velocity.
    """

    # Calculate binned statistics of velocity as a function of position.
    avg_vel, edges, _ = binned_statistic(norm_pos, vel, bins=nbins)

    # Calculate centers of each bin.
    centers = [(edges[i + 1] + edges[i]) / 2 for i in range(len(edges) - 1)]

    # Plot scatter plot with x-axis labeled "Position", y-axis labeled "Velocity",
    # and title "Position vs Velocity Scatterplot".
    plt.plot(centers, avg_vel)
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    # plt.title('Position vs Velocity Scatterplot')

    # Display the scatter plot.
    plt.show()


import numpy as np
import pandas as pd


def computed_sliced_matrix(trial_matrix, vel, t_on, t_off):
    """
    Computes a 2D array (trial x timepoints) of velocity values for a given time window around a sound onset.
    for each trial in a trial_matrix dataframe.

    PARAMETERS:
    trial_matrix (pandas.DataFrame): a dataframe containing the trial data.
    vel (numpy.ndarray): a 1D array of velocity values.
    t_on (int): the number of milliseconds before the sound onset to include in the velocity timecourse.
    t_off (int): the number of milliseconds after the sound onset to include in the velocity timecourse.

    RETURNS:
    vel_matrix (numpy.ndarray): a 2D array of velocity values for each trial, with shape (number of trials, t_on + t_off).
    """

    # Initialize the velocity matrix and count variable.
    vel_matrix = np.zeros((len(trial_matrix), t_on + t_off))
    count = 0

    # Cycle over the trials using integer indexing.
    for i in range(len(trial_matrix)):
        # Get the current trial row.
        row = trial_matrix.iloc[i]

        # Check if the row contains valid data.
        if not np.isnan(row["sound_onset"]):
            onset = row["sound_onset"].astype(int) - t_on  # 2 seconds before.
            offset = row["sound_onset"].astype(int) + t_off  # 2 seconds after.
            trial_vel = vel[onset:offset]

            # Add the trial's velocity timecourse to the velocity matrix.
            vel_matrix[count, :] = trial_vel
            count += 1

    # Truncate the velocity matrix to remove rows with NaN values.
    vel_matrix = vel_matrix[:count, :]

    return vel_matrix


import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt


def avg_std_sem_velocity(vel_matrix, t_on, t_off):
    """
    Computes the average, standard deviation, and standard error of the mean of velocity from a velocity matrix,
    and plots the average velocity over time with the fill_between visible.

    PARAMETERS:
    vel_matrix (numpy.ndarray): A 2D numpy array where each row represents a different trial and each column
        represents a different time point.
    t_on (int): The duration of the time period before sound onset, in milliseconds.
    t_off (int): The duration of the time period after sound onset, in milliseconds.

    RETURNS:
    tuple: A tuple of 3 numpy arrays containing the average velocity, standard deviation, and standard error of
        the mean of velocity, respectively.
    """

    # Compute the average velocity across trials for each time point.
    avg_vel = np.mean(vel_matrix, axis=0)

    # Compute the standard deviation of velocity across trials for each time point.
    std_vel = np.std(vel_matrix, axis=0)

    # Compute the standard error of the mean of velocity across trials for each time point.
    sem_vel = sem(vel_matrix, axis=0)

    # Generate an array of time points in seconds.
    t = np.linspace(-t_on / 1000, t_off / 1000, t_on + t_off)

    # Plot the average velocity over time.
    plt.plot(t, avg_vel)

    # Fill the area between the upper and lower bounds of the standard error of the mean.
    plt.fill_between(t, avg_vel - sem_vel, avg_vel + sem_vel, alpha=0.5)

    # Draw a red vertical line at the time of sound onset.
    plt.axvline(x=0, c="r")

    # Show the plot.
    plt.show()

    # Return the average, standard deviation, and standard error of the mean of velocity as a tuple.
    return avg_vel, std_vel, sem_vel


from scipy.stats import ttest_ind


def ttest_speed_distribution(vel_matrix, t_on, alpha=0.05):
    """
    Computes the t-test for the speed distribution before and after the sound onset.

    PARAMETERS:
    vel_matrix (numpy.ndarray): a 2D array of velocity values for each trial, with shape (number of trials, t_on + t_off).
    alpha: can choose a significance level by passing a value for the alpha parameter when calling the function.

    RETURNS:
    t (float): the t-test statistic.
    p (float): the p-value.
    """

    # Slice vel_matrix at sound onset.
    vel_before = vel_matrix[
        :, :t_on
    ].flatten()  # Selects all rows and the first t_on columns of the vel_matrix, which corresponds to the velocity values before the sound onset.
    vel_after = vel_matrix[
        :, t_on:
    ].flatten()  # Selects all rows and the columns starting from the t_on-th column of the vel_matrix, which corresponds to the velocity values after the sound onset.

    # Compute t-test between vel_before and vel_after.
    t_stat, p_val = ttest_ind(vel_before, vel_after)

    # Check if the difference between the speed distributions before and after the sound onset is statistically significant.
    if p_val < alpha:
        # If p-value is less than the significance level, reject the null hypothesis and print message indicating statistical significance.
        print(
            "The difference between the speed distributions before and after the sound onset is statistically significant (p < {:.4f})".format(
                alpha
            )
        )
        # If p-value is greater than or equal to the significance level, fail to reject the null hypothesis and print message indicating no statistical significance.
    else:
        print(
            "There is no statistically significant difference between the speed distributions before and after the sound onset (p = {:.4f})".format(
                p_val
            )
        )

    return t_stat, p_val
