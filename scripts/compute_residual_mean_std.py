import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from meteolibre_model.dataset.dataset_mtg_lightning import MeteoLibreMapDataset
from tqdm import tqdm

def compute_residual_mean_std():
    """
    Computes the mean and standard deviation of the residual (sat_patch_data[0] - sat_patch_data[1]) for each channel,
    ignoring pixels where either time step has -10000 values.
    """
    # Load the local dataset
    localrepo = "/workspace/dataset"
    dataset = MeteoLibreMapDataset(localrepo=localrepo)

    # Use a smaller subset for quick estimation, or iterate over the whole dataset
    # for a more accurate result. Set num_samples to None to iterate over everything.
    num_samples = 20000  # Or None

    print(f"Calculating residual mean and std over {num_samples or len(dataset)} samples, ignoring invalid pixels...")

    # Initialize accumulators for residual data
    res_total_sum = None
    res_total_sum_sq = None
    res_total_count = None

    for i, sample in enumerate(tqdm(dataset, total=num_samples)):
        if num_samples is not None and i >= num_samples:
            break

        # Process sat data for residuals
        sat_patch_data = sample["sat_patch_data"].numpy().astype(np.float32)
        if sat_patch_data.shape[0] < 2:
            print(f"Warning: Sample {i} has fewer than 2 time steps, skipping.")
            continue

        t0 = sat_patch_data[0]  # (C, H, W)
        t1 = sat_patch_data[1]  # (C, H, W)

        residual_data = t0 - t1  # (C, H, W)

        # Mask where either t0 or t1 is -10000
        mask_t0 = np.equal(t0, -10000.0)
        mask_t1 = np.equal(t1, -10000.0)
        mask = np.logical_or(mask_t0, mask_t1)
        masked_residual = np.ma.masked_where(mask, residual_data)

        if res_total_sum is None:
            num_channels = residual_data.shape[0]
            res_total_sum = np.zeros(num_channels, dtype=np.float64)
            res_total_sum_sq = np.zeros(num_channels, dtype=np.float64)
            res_total_count = np.zeros(num_channels, dtype=np.int64)

        res_total_sum += masked_residual.sum(axis=(1, 2)).filled(0)
        res_total_sum_sq += np.ma.power(masked_residual, 2).sum(axis=(1, 2)).filled(0)
        res_total_count += masked_residual.count(axis=(1, 2))

    # Calculate mean and std for residuals
    assert res_total_sum is not None
    res_mean = np.divide(res_total_sum, res_total_count, out=np.zeros_like(res_total_sum), where=res_total_count != 0)
    res_variance = np.divide(res_total_sum_sq, res_total_count, out=np.zeros_like(res_total_sum_sq), where=res_total_count != 0) - np.square(res_mean)
    res_std = np.sqrt(res_variance)

    print("\nResidual Mean for each channel:")
    print(res_mean)
    print("\nResidual Standard deviation for each channel:")
    print(res_std)

if __name__ == "__main__":
    compute_residual_mean_std()