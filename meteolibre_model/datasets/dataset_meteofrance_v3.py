"""
This is the module to handle the dataloading from the huggingface dataset
"""

from datetime import datetime
import pandas as pd
import os
import json

import numpy as np
from numpy.random import default_rng

from datasets import load_dataset
import torch

STATISTIC_SATELLITE_MEAN = torch.tensor([0.1108, 0.1524, 0.0602, 0.1168])
STATISTIC_SATELLITE_STD = torch.tensor([0.0839, 0.1506, 0.0383, 0.0990])


STATISTIC_RADAR_MEAN = 0.008580314926803112
STATISTIC_RADAR_STD = 0.1423337459564209


def read_record(record):
    """Reads a single line from the dataset."""

    satellite_data = np.frombuffer(record['satellite'], dtype=np.int16).reshape(9, 4, 256, 256) / 65536
    satellite_data = satellite_data.astype(np.float32)

    hour = record['hour']
    minute = record['minute']
    datetime = record['datetime']
    radar_data =  np.frombuffer(record['radar'], dtype=np.float32).reshape(9, 256, 256).copy()
    gs_data =  np.frombuffer(record['groundstation'], dtype=np.float32).reshape(9, 256, 256, 7).copy()
    height_data =  np.frombuffer(record['ground_height'], dtype=np.float32).reshape(256, 256).copy()
    landcover_data =  np.frombuffer(record['landcover'], dtype=np.float32).reshape(256, 256, 4).copy()

    gs_data = np.where(gs_data == -100, -4, gs_data)
    gs_data = np.where(np.isnan(gs_data), -4, gs_data)

    return {
        "radar": radar_data,  # The NumPy array itself
        "groundstation": gs_data,  # The NumPy array itself
        "ground_height": height_data,  # The NumPy array itself
        "satellite": satellite_data,
        "landcover": landcover_data,
        "hour": hour / 24.,  # The scalar value
        "minute": minute / 60.,  # The scalar value
        # "time_radar": time_radar_back,  # The scalar value
        # "datetime": datetime,  # The scalar value
        # "id": id,  # The scalar value
    }

class MeteoLibreDatasetHF(torch.utils.data.IterableDataset):
    def __init__(self, localrepo=None, hf_repo="meteolibre-dev/mtg_meteofrance_256"):
        super().__init__()

        if localrepo is None:
            self.remote = True
            self.ds = load_dataset(hf_repo, streaming=True)
        else:
            self.remote = False
            self.ds = load_dataset(localrepo, data_files = localrepo + "/data/*.parquet", streaming=True)

    def __iter__(self):
        for row in self.ds["train"]:
            
            data = read_record(row)
            yield data