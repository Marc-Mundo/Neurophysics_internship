import matplotlib.pyplot as plt
import os


def lick_counter(trial_data, b_data):
    """
    Calculates the number of licks within a specified time frame and computes the percentage of reward-related licks.

    Args:
        trial_data (dict): Dictionary containing trial data with keys 'reward_onset', 'tunnel2_offset', and 'env_onset'.
        b_data (dict): Dictionary containing lick data with the key 'lick_onsets'.

    Returns:
        tuple: A tuple containing the lick counter and the percentage of reward-related licks.

    """
    # Define the time frame during the trial in which the reward was presented and licks are part of the desired behaviour.
    lick_start = trial_data["reward_onset"]
    lick_end = trial_data["reward_onset"] + 2000

    # Numpy array of the number of licks during the lick time.
    licks = b_data["lick_onsets"]

    # Init the lick counter.
    lick_counter = 0

    # Iterate through each row in the dataframe.
    for i in range(len(lick_start)):
        # Check if the values fall within the range.
        for lick in licks:
            if lick_start[i] <= lick <= lick_end[i]:
                lick_counter += 1

    # Compute the fraction of licks that happen at the reward_onset.
    reward_licks = lick_counter / len(licks) * 100

    return lick_counter, reward_licks


def lick_eventplot(trial_data, b_data, session_path, save_folder):
    """
    Generate an eventplot of licks per trial based on trial_data and b_data
    and save the plot as an image file.

    Args:
        trial_data (dict): Dictionary containing trial data.
        b_data (dict): Dictionary containing lick data.
        session_path (WindowsPath): Path of the session.
        save_folder (str): Path of the folder to save the image to.
    Returns:
        None: Displays the eventplot.

    """
    # Calculating some values needed to find the number of licks.
    trial_duration = (
        trial_data["tunnel2_offset"] - trial_data["env_onset"]
    )  # full trial duration.

    # Define the time frame during the trial in which the reward was presented and licks are part of the desired behaviour.
    lick_start = trial_data["reward_onset"]
    lick_end = trial_data["reward_onset"] + 2000

    # Numpy array of the number of licks during the lick time.
    licks = b_data["lick_onsets"]

    # Init the lick counter.
    lick_counter = 0

    # Init list of reward licks.
    reward_licks_list = []

    # Iterate through each row in the dataframe.
    for i in range(len(lick_start)):
        # Init list for adding the licks at the reward zone onset.
        trial_reward_licks = []

        # Check if the values fall within the range.
        for lick in licks:
            if lick_start[i] <= lick <= lick_end[i]:
                lick_counter += 1
                relative_lick_time = (
                    lick - lick_start[i]
                )  # Lick time with respect to reward onset.
                trial_reward_licks.append(relative_lick_time)

        # Append the list of licks of the trial to the general list, the general list will be a list of lists.
        reward_licks_list.append(trial_reward_licks)

    print(lick_counter)

    # Compute the fraction of licks that happen at the reward_onset.
    reward_licks = lick_counter / len(licks)
    print(reward_licks)

    # Create the eventplot.
    plt.eventplot(reward_licks_list, lineoffsets=1, linelengths=0.5)

    # Add axis labels and title.
    plt.xlabel("Time (frames), w.r.t reward onset")
    plt.ylabel("Trial")
    plt.title("Eventplot of Licks per Trial")

    # Get the session number from the session path
    session_number = os.path.basename(session_path)

    # Set the image file name
    image_name = f"{session_number}_lick_eventplot.png"

    # Set the complete save path including the folder and image name
    save_path = os.path.join(save_folder, image_name)

    # Save the plot as an image file
    plt.savefig(save_path)

    # Show the plot.
    plt.show()
