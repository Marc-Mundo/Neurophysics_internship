import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


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


def position_event_histogram(b_data, trial_data, session_path, save_folder):
    """
    Plot a histogram of the normalized position with markers for specific events:
    Reward zone onset, Tunnel1 onset, Sound onset.

    Parameters:
        b_data (pandas.DataFrame): A DataFrame containing the behavior data.
        trial_data (pandas.DataFrame): A DataFrame containing the trial data.
        session_path (str or pathlib.Path): The path to the session data.
        save_folder (str or pathlib.Path): The folder path to save the histogram image.

    Returns:
        None. Displays the histogram plot and saves it as an image file.

    Example usage:
        position_event_histogram(b_data, trial_data, session_path, save_folder)
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

    # Get the session number from the session path
    session_number = os.path.basename(session_path)

    # Set the image file name
    image_name = f"{session_number}_position_event_histogram.png"

    # Set the complete save path including the folder and image name
    save_path = os.path.join(save_folder, image_name)

    # Save the plot as an image file
    plt.savefig(save_path)

    # Display the plot
    plt.show()
