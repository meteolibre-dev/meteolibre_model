"""
This is the module to handle the dataloading from the huggingface dataset
"""

from datetime import datetime
import pandas as pd
import os
import json
import glob
from collections import OrderedDict

import numpy as np
from numpy.random import default_rng
import random

from datasets import load_dataset
from torch.utils.data import DataLoader, get_worker_info
import torch
import torch.distributed as dist

STATISTIC_SATELLITE_MEAN = torch.tensor([0.1108, 0.1524, 0.0602, 0.1168])
STATISTIC_SATELLITE_STD = torch.tensor([0.0839, 0.1506, 0.0383, 0.0990])


STATISTIC_RADAR_MEAN = 0.008580314926803112
STATISTIC_RADAR_STD = 0.1423337459564209


class MeteoLibreMapDataset(torch.utils.data.Dataset):
    """
    An advanced map-style Dataset that improves upon the cached version by
    shuffling the order of Parquet files differently for each DataLoader worker.

    This ensures that workers are likely to request different files at the
    same time, which can improve data loading randomness and potentially reduce
    contention for the same files if they were on a shared network drive.

    Args:
        localrepo (str): The path to the local repository.
        records_per_file (int): The fixed number of records in each Parquet file.
        cache_size (int): The number of DataFrames to keep in the in-memory cache.
        seed (int): A base seed used to ensure reproducible shuffling across runs.
                    Each worker's shuffle seed will be `seed + worker_id`.
    """
    def __init__(self, localrepo: str, records_per_file: int = 50, cache_size: int = 8, seed: int = 42):
        super().__init__()
        self.localrepo = localrepo
        self.records_per_file = records_per_file
        self.cache_size = cache_size
        self.seed = seed # Store base seed

        # Find all parquet files. We start with a sorted list for consistency.
        data_path = os.path.join(self.localrepo, "data", "*.parquet")
        self.base_file_paths = sorted(glob.glob(data_path))
        
        # we remove the last one 
        self.base_file_paths = self.base_file_paths[:-1]


        if not self.base_file_paths:
            raise FileNotFoundError(f"No Parquet files found at '{data_path}'.")

        self.file_paths = self.base_file_paths
            
        # Initialize an LRU cache for this worker
        self.cache = OrderedDict()

        # Calculate the total number of records
        self.total_records = len(self.file_paths) * self.records_per_file

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

    def _preprocess(self, record: pd.Series) -> dict:
        patch_data = np.frombuffer(record["data"], dtype=record["dtype"]).reshape(record["shape"])

        # TODO complete with context value later

        return {
            "patch_data": patch_data
        }

    def __getitem__(self, index: int) -> dict:
        if not getattr(self, 'worker_initialized', False):
            worker_info = get_worker_info()
            if worker_info is not None:
                # We are in a worker process. Shuffle the files based on worker ID.
                worker_id = worker_info.id
                #print("worker_id", worker_id)
                
                # Get rank of the process to ensure different shuffling across GPUs
                rank = 0
                if dist.is_available() and dist.is_initialized():
                    rank = dist.get_rank()
                    
                #print("rank", rank)

                # Create a generator with a seed unique to this worker and the base seed
                g = torch.Generator()
                
                seed = self.seed + worker_id + rank * worker_info.num_workers + random.randint(0, 1000)# + round(datetime.now().timestamp() * 1000)
                g.manual_seed(seed)
                
                # Get a random permutation of indices and apply it to the file list
                perm = torch.randperm(len(self.base_file_paths), generator=g).tolist()
                self.file_paths = [self.base_file_paths[i] for i in perm]
            self.worker_initialized = True

        if index < 0 or index >= self.total_records:
            raise IndexError(f"Index {index} out of range for dataset with size {self.total_records}")

        file_index = index // self.records_per_file
        row_index_in_file = index % self.records_per_file
        data_df = self._get_dataframe(file_index)
        try:
            record = data_df.iloc[row_index_in_file]
            return self._preprocess(record)
        except Exception as e:
            print(f"Bad indexing for index {index}, file_index {file_index}, row_index {row_index_in_file}. Error: {e}")
            return self.__getitem__((index + 1) % self.total_records)