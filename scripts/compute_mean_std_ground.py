import numpy as np
import torch
from tqdm import tqdm
import argparse
import einops
from meteolibre_model.dataset.dataset_mtg_meteofrance import MeteoLibreMapDataset

def compute_mean_std_ground(localrepo: str, num_samples: int = 10000):
    """
    Computes the mean and standard deviation of the ground_station_data channels,
    ignoring -10.0 values (fill value for missing data).
    """
    # Load the dataset
    dataset = MeteoLibreMapDataset(localrepo=localrepo)

    # Initialize accumulators for 7 KPIs
    num_kpis = 7
    total_sum = np.zeros(num_kpis, dtype=np.float64)
    total_sum_sq = np.zeros(num_kpis, dtype=np.float64)
    total_count = np.zeros(num_kpis, dtype=np.int64)

    print(f"Calculating mean and std over {num_samples} samples, ignoring -10.0...")

    for i in tqdm(range(min(num_samples, len(dataset)))):
        sample = dataset[i]
        ground_data = sample["ground_station_data"].numpy()  # (5, 7, 128, 128)

        # Flatten temporal and spatial dimensions, keep KPI dimension
        # Shape: (7, 5*128*128)
        ground_flat = einops.rearrange(ground_data, "time kpi h w -> kpi (time h w)")

        # Mask out the -10.0 values
        masked_data = np.ma.masked_equal(ground_flat, -10.0)

        # Update accumulators per KPI
        total_sum += masked_data.sum(axis=1).filled(0)
        total_sum_sq += np.ma.power(masked_data, 2).sum(axis=1).filled(0)
        total_count += masked_data.count(axis=1)

    # Calculate mean and std
    mean = np.divide(total_sum, total_count, out=np.zeros_like(total_sum), where=total_count != 0)
    variance = np.divide(total_sum_sq, total_count, out=np.zeros_like(total_sum_sq), where=total_count != 0) - np.square(mean)
    std = np.sqrt(variance)

    print("\nMean for each KPI:")
    print(mean)
    print("\nStandard deviation for each KPI:")
    print(std)

    return mean, std

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mean and std for ground station data")
    parser.add_argument("localrepo", type=str, help="Path to the local repository containing data")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to use for computation")
    args = parser.parse_args()

    compute_mean_std_ground(args.localrepo, args.num_samples)