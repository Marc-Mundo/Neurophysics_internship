import glob
from pathlib import Path
import pickle
import numpy as np


def get_files_in_data_path(data_path):
    """
    Retrieves a list of files within a specified data path.

    Args:
        data_path (Path): The path to the data directory.

    Returns:
        list: A list of file paths within the data directory.
    """
    # Use glob to get a list of files in the data path
    files = glob.glob(str(data_path) + "\\*")
    return files


def visualize_data_folders(data_folders):
    """
    Accesses and visualizes the files within the specified data folders.

    Args:
        data_folders (list): A list of strings representing the paths to the data folders.

    Returns:
        None
    """
    # Create a list to store all the data paths
    all_data_paths = []

    # Iterate over each data folder
    for folder in data_folders:
        data_folder = Path(folder)
        all_data_paths.append(data_folder)

    # Iterate over each data path
    for data_path in all_data_paths:
        # Use glob to get a list of files in the data path
        files = glob.glob(str(data_path) + "\\*")

        # Visualize the files
        for file in files:
            print(file)


def ndata_contents(file_path):
    """
    Prints the contents of a neural data pickle file.

    Args:
        file_path (str): The file path of the neural data pickle file.

    Returns:
        None
    """
    with open(file_path, "rb") as file:
        n_data = pickle.load(file)

    for k in n_data.keys():
        if type(n_data[k]) == np.ndarray:
            print(f"{k} -> array with shape: {n_data[k].shape}")
        else:
            print(f"{k} -> list with length: {len(n_data[k])}")
