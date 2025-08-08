"""
This is the module to handle the dataloading from the huggingface dataset
"""

from datetime import datetime
import pandas as pd
import os
import json

import numpy as np
from numpy.random import default_rng

import torch

STATISTIC_SATELLITE_MEAN = torch.tensor([0.1108, 0.1524, 0.0602, 0.1168])
STATISTIC_SATELLITE_STD = torch.tensor([0.0839, 0.1506, 0.0383, 0.0990])


STATISTIC_RADAR_MEAN = 0.008580314926803112
STATISTIC_RADAR_STD = 0.1423337459564209

def load_jsonl_to_dataframe(file_path):
    """
    Reads a .jsonl file line by line and loads it into a pandas DataFrame.

    Args:
        file_path (str): The path to the .jsonl file.

    Returns:
        pandas.DataFrame: A DataFrame containing the data from the file,
                          or None if the file is not found or is empty.
    """
    data_list = []
    try:
        # Open the file for reading. The 'with' statement ensures the file is properly closed.
        with open(file_path, 'r') as f:
            # Iterate over each line in the file.
            for line in f:
                # Each line is a JSON string, so we parse it into a dictionary
                # and append it to our list. We use strip() to remove any potential trailing newline characters.
                if line.strip():  # Ensure the line is not empty
                    try:
                        data_list.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Skipping line due to JSON decoding error: {e}")
                        print(f"Problematic line: {line.strip()}")

        # Create a pandas DataFrame from the list of dictionaries if data was loaded.
        if data_list:
            df = pd.DataFrame(data_list)
            return df
        else:
            print("No data was loaded. The file might be empty or all lines had errors.")
            return None

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please make sure the file exists and the path is correct.")
        return None


def read_record(record):
    """Reads a single line from the dataset."""
    radar_file = record["radar_file_path"]

    gs = record["groundstation_file_path"]
    satellite = record["satellite_file_path"]

    height = record["ground_height_file_path"]
    landcover = record["landcover_file_path"]

    hour = record["hour"]
    minutes = record["minute"]
    time_radar_back = record["time_radar"]
    datetime = record["datetime"]
    id = record["id"]

    # Load the all the npz files
    npz_paths = [radar_file, gs, height]
    # Check if all files exist
    for npz_path in npz_paths:
        if not os.path.exists(npz_path):
            print(f"File not found: {npz_path}")
            continue
    # Load the npz files
    try:
        radar_data = np.load(radar_file)[
            "arr_0"
        ]  # Assuming the data is stored under 'arr_0'
        gs_data = np.load(gs)["arr_0"]
        height_data = np.load(height)["arr_0"]
        landcover_data = np.load(landcover)["arr_0"]
        satellite_data = np.load(satellite)["arr_0"] / 65536
    except Exception as e:
        print(
            f"Error loading data from {radar_data}, {gs_data}, {height}: {e}"
        )
        raise ("Error")

    gs_data = np.where(gs_data == -100, -4, gs_data)
    gs_data = np.where(np.isnan(gs_data), -4, gs_data)

    return {
        "radar": radar_data,  # The NumPy array itself
        "groundstation": gs_data,  # The NumPy array itself
        "ground_height": height_data,  # The NumPy array itself
        "satellite": satellite_data,
        "landcover": landcover_data,
        "hour": hour / 24.,  # The scalar value
        "minute": minutes / 60.,  # The scalar value
        # "time_radar": time_radar_back,  # The scalar value
        # "datetime": datetime,  # The scalar value
        # "id": id,  # The scalar value
    }

class MeteoLibreDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, directory):
        super().__init__()

        if isinstance(directory, str):
            directories = [directory]
        else:
            directories = directory
        
        self.directories = directories

        all_indexes = []
        for d in directories:
            json_path = os.path.join(d, "index.json")
            index_data = load_jsonl_to_dataframe(json_path)

            if index_data is not None:
                for column in index_data.columns:
                    if "file" in column:
                        index_data[column] = d + index_data[column].str[19:]
                all_indexes.append(index_data)

        if all_indexes:
            self.index_data = pd.concat(all_indexes, ignore_index=True)
        else:
            self.index_data = pd.DataFrame()
            
        self.index_data.head()
        
        # filter not coherent element (not equal spacing)
        element = []
        for value in self.index_data["time_radar"].values:
            if value[0] != 2. or  value[-1] != -2.:
                element.append(False)
            else:
                element.append(True)
            
        self.index_data = self.index_data[element].reset_index()

    def __len__(self):
        return len(self.index_data)

    def __getitem__(self, item):
        row = self.index_data.iloc[item]

        # read data
        data = read_record(row)

        return data
