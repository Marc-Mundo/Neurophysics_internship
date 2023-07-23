import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import binned_statistic
from scipy.stats import sem
from scipy.stats import ttest_ind
import os


def compute_velocity(
    b_data,
    session_path,
    save_folder,
    dt=1.0 / 1000.0,
    pos_sigma=2,
    vel_sigma=50,
    vel_win=10000,
    vel_smooth=100,
    show_plot=True,
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
    plt.ylabel("Velocity (cm/s)")
    # plt.title('Velocity plot')

    # Get the session number and parent folder from the session path
    session_number = os.path.basename(session_path)
    parent_folder = os.path.basename(os.path.dirname(session_path))

    # Concatenate the parent folder and session number
    session_name = f"{parent_folder}_{session_number}"

    # Set the image file name
    image_name = f"{session_name}_velocity_plot.png"

    # Set the complete save path including the folder and image name
    save_path = os.path.join(save_folder, image_name)

    # Save the plot as an image file
    plt.savefig(save_path)

    # If show_plot is True, then display the plot
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the plot if show_plot is False, so it's not shown on the screen.

    # Return the velocity data.
    return vel


def pos_vel_histogram(
    norm_pos, vel, session_path, save_folder, nbins=50, show_plot=True
):
    """
    Create a histogram plot of the relationship between position and velocity
    using binned statistics.

    PARAMETERS:
    norm_pos : array-like
            The normalized positions.
    vel : array-like
            The velocities.
    nbins : int, optional
            The number of bins to use for the histogram plot (default=50).

    RETURNS:
    A histogram with graph between datapoints to visualize the relationship between position and velocity.
    """
    # Calculate binned statistics of velocity as a function of position.
    avg_vel, edges, _ = binned_statistic(norm_pos, vel, bins=nbins)

    # Calculate centers of each bin.
    centers = [(edges[i + 1] + edges[i]) / 2 for i in range(len(edges) - 1)]

    # Plot scatter plot with x-axis labeled "Position", y-axis labeled "Velocity",
    # and title "Position vs Velocity Scatterplot".
    plt.plot(centers, avg_vel)
    plt.xlabel("Normalized Position")
    plt.ylabel("Velocity cm/s")
    # plt.title('Position vs Velocity Scatterplot')

    # Get the session number and parent folder from the session path
    session_number = os.path.basename(session_path)
    parent_folder = os.path.basename(os.path.dirname(session_path))

    # Concatenate the parent folder and session number
    session_name = f"{parent_folder}_{session_number}"

    # Set the image file name
    image_name = f"{session_name}_pos_vel_histogram_plot.png"

    # Set the complete save path including the folder and image name
    save_path = os.path.join(save_folder, image_name)

    # Save the plot as an image file
    plt.savefig(save_path)

    # If show_plot is True, then display the plot
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the plot if show_plot is False, so it's not shown on the screen.


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
    max_trials = len(trial_matrix)

    # Initialize the velocity matrix and count variable.
    vel_matrix = np.zeros((len(trial_matrix), t_on + t_off))
    count = 0

    # Cycle over the trials using integer indexing.
    for i in range(max_trials):
        # Get the current trial row.
        row = trial_matrix.iloc[i]

        # Check if the row contains valid data.
        if not np.isnan(row["sound_onset"]):
            onset = row["sound_onset"].astype(int) - t_on  # 2 seconds before.
            offset = row["sound_onset"].astype(int) + t_off  # 2 seconds after.
            trial_vel = vel[onset:offset]

            # Check if trial_vel is not empty before adding to vel_matrix
            if trial_vel.size > 0:
                vel_matrix[count, : trial_vel.size] = trial_vel

            # Add the trial's velocity timecourse to the velocity matrix.
            count += 1

    # After the loop, you might have some empty rows in vel_matrix due to skipped trials.
    # You can remove these empty rows to have a clean vel_matrix with only valid trials.
    vel_matrix = vel_matrix[:count]

    return vel_matrix


def avg_std_sem_velocity(
    vel_matrix, t_on, t_off, session_path, save_folder, show_plot=True
):
    """
    Computes the average, standard deviation, and standard error of the mean of velocity from a velocity matrix,
    and plots the average velocity over time with the fill_between visible.
    Saves the plot to the specified save_folder

    PARAMETERS:
    vel_matrix (numpy.ndarray): A 2D numpy array where each row represents a different trial and each column
        represents a different time point.
    t_on (int): The duration of the time period before sound onset, in milliseconds.
    t_off (int): The duration of the time period after sound onset, in milliseconds.
    session_path (WindowsPath): Path of the session.
    save_folder (str): Path of the folder to save the image to.

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
    plt.axvline(x=0, c="r", label="Sound Onset")

    # Add legend
    plt.legend()

    # Add x and y labels
    plt.xlabel("Time(s)")
    plt.ylabel("Velocity (cm/s)")

    # Get the session number and parent folder from the session path
    session_number = os.path.basename(session_path)
    parent_folder = os.path.basename(os.path.dirname(session_path))

    # Concatenate the parent folder and session number
    session_name = f"{parent_folder}_{session_number}"

    # Set the image file name
    image_name = f"{session_name}_average_velocity_plot.png"

    # Set the complete save path including the folder and image name
    save_path = os.path.join(save_folder, image_name)

    # Save the plot as an image file
    plt.savefig(save_path)

    # If show_plot is True, then display the plot
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the plot if show_plot is False, so it's not shown on the screen.

    # Return the average, standard deviation, and standard error of the mean of velocity as a tuple.
    return avg_vel, std_vel, sem_vel


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


def plot_combined_average_velocity(
    all_vel_matrices, t_on, t_off, save_folder, show_plot=True
):
    """
    Plots the combined average velocity over time with the fill_between visible.
    Saves the plot to the specified save_folder if show_plot is True.

    PARAMETERS:
    all_vel_matrices (list): A list of 2D numpy arrays where each array represents velocity data from a different session.
    t_on (int): The duration of the time period before sound onset, in milliseconds.
    t_off (int): The duration of the time period after sound onset, in milliseconds.
    save_folder (str): Path of the folder to save the image to.
    show_plot (bool, optional): Whether to display the plot. Defaults to True.

    RETURNS:
    tuple: A tuple of three numpy arrays containing the combined average velocity, standard deviation,
           and standard error of the mean of velocity, respectively.
    """
    # Combine the velocity matrices from all sessions
    combined_vel_matrix = np.concatenate(all_vel_matrices, axis=0)

    # Compute the average velocity across trials for each time point.
    avg_vel_combined = np.mean(combined_vel_matrix, axis=0)

    # Compute the standard deviation of velocity across trials for each time point.
    std_vel_combined = np.std(combined_vel_matrix, axis=0)

    # Compute the standard error of the mean of velocity across trials for each time point.
    sem_vel_combined = np.std(combined_vel_matrix, axis=0) / np.sqrt(
        combined_vel_matrix.shape[0]
    )

    # Generate an array of time points in seconds.
    t = np.linspace(-t_on / 1000, t_off / 1000, t_on + t_off)

    # Plot the combined average velocity over time.
    plt.plot(t, avg_vel_combined)

    # Fill the area between the upper and lower bounds of the standard error of the mean.
    plt.fill_between(
        t,
        avg_vel_combined - sem_vel_combined,
        avg_vel_combined + sem_vel_combined,
        alpha=0.5,
    )

    # Draw a red vertical line at the time of sound onset.
    plt.axvline(x=0, c="r", label="Sound Onset")

    # Add legend
    plt.legend()

    # Add x and y labels
    plt.xlabel("Time(s)")
    plt.ylabel("Velocity (cm/s)")

    # Set the image file name
    image_name = "average_velocity_plot_combined.png"

    # Set the complete save path including the folder and image name
    save_path = os.path.join(save_folder, image_name)

    # Save the plot as an image file if show_plot is True
    if show_plot:
        plt.savefig(save_path)
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()  # Close the plot if show_plot is False, so it's not shown on the screen.

    # Return the average velocity, standard deviation, and standard error of the mean of velocity as a tuple.
    return avg_vel_combined, std_vel_combined, sem_vel_combined
