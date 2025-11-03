"""
This is the module to handle the dataloading from the huggingface dataset

This is the dataloader for a 7-frame version of the meteolibre-dev/weather_mtg_europe_lightning_128_1km huggingface dataset
"""


from datetime import datetime, timedelta
import pandas as pd
import os
import glob
from collections import OrderedDict
import bisect
import ast

import numpy as np
import scipy.sparse as sp
import random
import pyarrow.parquet as pq

from torch.utils.data import get_worker_info
import torch
import torch.distributed as dist


from suncalc import get_position


class MeteoLibreMapDataset7Frames(torch.utils.data.Dataset):
    """
    A map-style Dataset for loading 7-frame sequences.

    This class is adapted from the 5-frame version. It loads sequences of 7 frames
    and computes the solar position for the 3 future frames that will be predicted.
    """

    def __init__(self, localrepo: str, cache_size: int = 8, seed: int = 42, nb_temporal: int = 7):
        super().__init__()
        self.localrepo = localrepo
        self.cache_size = cache_size
        self.seed = seed  # Store base seed
        self.nb_temporal = nb_temporal

        # Find all parquet files. We start with a sorted list for consistency.
        data_path = os.path.join(self.localrepo, "data", "*.parquet")
        self.base_file_paths = sorted(glob.glob(data_path))

        if not self.base_file_paths:
            raise FileNotFoundError(f"No Parquet files found at '{data_path}'.")

        self.file_paths = self.base_file_paths

        # Initialize an LRU cache for this worker
        self.cache = OrderedDict()

        # Calculate the total number of records by inspecting each file
        self.records_per_file_list = [
            sum(p.count_rows() for p in pq.ParquetDataset(fp).fragments)
            for fp in self.base_file_paths
        ]
        self.total_records = sum(self.records_per_file_list)

        # Create a cumulative sum of records to quickly find the file for an index
        self.cumulative_records = np.cumsum(
            [0] + self.records_per_file_list[:-1]
        ).tolist()

    def __len__(self) -> int:
        return self.total_records

    def _get_dataframe(self, file_index: int) -> pd.DataFrame:
        if file_index in self.cache:
            self.cache.move_to_end(file_index)
            return self.cache[file_index]

        file_path = self.file_paths[file_index]
        data_df = pd.read_parquet(file_path)
        self.cache[file_index] = data_df

        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return data_df

    def _preprocess(self, date, record: pd.Series) -> dict:
        # satellite data
        sat_patch_data = (
            np.frombuffer(record["sat_data"], dtype=record["sat_dtype"])
            .reshape(record["sat_shape"])
            .copy()
        )

        lightning_patch_data = (
            np.frombuffer(record["lightning_data"], dtype=record["lightning_dtype"])
            .reshape(record["lightning_shape"])
            .copy()
        )

        # Ensure we have exactly self.nb_temporal frames
        if record["sat_shape"][0] > self.nb_temporal:
            sat_patch_data = sat_patch_data[:self.nb_temporal, :, :, :]

        if record["sat_shape"][0] < self.nb_temporal:
            return self.__getitem__(random.randint(0, self.__len__()))

        if record["lightning_shape"][0] > self.nb_temporal:
            lightning_patch_data = lightning_patch_data[:self.nb_temporal, :, :, :]

        # Retrieve longitude and latitude for sun position calculation
        long = record["lon"] # longitude
        lat = record["lat"] # latitude

        # Calculate sun positions for frames 5, 6, and 7 (0-indexed 4, 5, 6)
        # Timestamps are relative to the 'date' of the first frame (10 min resolution)
        date_frame5 = date + timedelta(minutes=40)
        date_frame6 = date + timedelta(minutes=50)
        date_frame7 = date + timedelta(minutes=60)

        pos5 = get_position(date_frame5, long, lat)
        pos6 = get_position(date_frame6, long, lat)
        pos7 = get_position(date_frame7, long, lat)

        lat_norm = lat / 10.0

        spatial_position_data = [
            [pos5["azimuth"], pos5["altitude"], lat_norm],
            [pos6["azimuth"], pos6["altitude"], lat_norm],
            [pos7["azimuth"], pos7["altitude"], lat_norm],
        ]

        batch_dict = {
            "sat_patch_data": torch.from_numpy(sat_patch_data),
            "lightning_patch_data": torch.from_numpy(lightning_patch_data),
            "spatial_position": torch.tensor(spatial_position_data),
        }

        return batch_dict

    def __getitem__(self, index: int) -> dict:
        if not getattr(self, "worker_initialized", False):
            worker_info = get_worker_info()
            if worker_info is not None:
                worker_id = worker_info.id
                rank = 0
                if dist.is_available() and dist.is_initialized():
                    rank = dist.get_rank()

                g = torch.Generator()
                seed = (
                    self.seed
                    + worker_id
                    + rank * worker_info.num_workers
                )
                g.manual_seed(seed)

                perm = torch.randperm(len(self.base_file_paths), generator=g).tolist()
                self.file_paths = [self.base_file_paths[i] for i in perm]

                self.records_per_file_list = [self.records_per_file_list[i] for i in perm]
                self.cumulative_records = np.cumsum([0] + self.records_per_file_list[:-1]).tolist()

            self.worker_initialized = True

        if index < 0 or index >= self.total_records:
            raise IndexError(
                f"Index {index} out of range for dataset with size {self.total_records}"
            )

        file_index = bisect.bisect_right(self.cumulative_records, index) - 1
        row_index_in_file = index - self.cumulative_records[file_index]

        file_path = self.file_paths[file_index]
        filename = os.path.basename(file_path)
        date_str = filename.split("_")[:2]
        date_str = "_".join(date_str)

        date = datetime.strptime(date_str, "%Y-%m-%d_%H-%M")

        data_df = self._get_dataframe(file_index)

        try:
            record = data_df.iloc[row_index_in_file]
            return self._preprocess(date, record)
        except Exception as e:
            print(
                f"Bad indexing for index {index}, file_index {file_index}, row_index {row_index_in_file}. Error: {e}"
            )
            return self.__getitem__((index + 64) % self.total_records)
