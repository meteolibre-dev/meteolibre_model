import numpy as np
from datasets import load_dataset
from tqdm import tqdm

def compute_mean_std():
    """
    Computes the mean and standard deviation of the dataset channels, ignoring -10000 values.
    """
    # Load the dataset
    ds = load_dataset("meteolibre-dev/mtg_europe_64", streaming=True, split="train")

    # Initialize accumulators
    num_channels = 12
    total_sum = np.zeros(num_channels, dtype=np.float64)
    total_sum_sq = np.zeros(num_channels, dtype=np.float64)
    total_count = np.zeros(num_channels, dtype=np.int64)

    # Use a smaller subset for quick estimation, or iterate over the whole dataset
    # for a more accurate result. Set num_samples to None to iterate over everything.
    num_samples = 20000  # Or None

    print(f"Calculating mean and std over {num_samples or 'all'} samples, ignoring -10000...")

    for i, example in enumerate(tqdm(ds, total=num_samples)):
        if num_samples is not None and i >= num_samples:
            break

        # Get the byte string and reshape
        patch_data = np.frombuffer(
            example["data"], dtype=np.dtype(example["dtype"])
        ).reshape(example["shape"])

        # Ensure patch_data is float64 for precision in sums
        patch_data = patch_data.astype(np.float32)

        # Mask out the -10000 values
        masked_patch_data = np.ma.masked_equal(patch_data, -10000.0)

        # Update accumulators
        # The shape is (6, 12, 64, 64), so we sum over axes 0, 2, and 3
        total_sum += masked_patch_data.sum(axis=(0, 2, 3)).filled(0)
        total_sum_sq += np.ma.power(masked_patch_data, 2).sum(axis=(0, 2, 3)).filled(0)

        # Number of valid (not masked) values per channel in this patch
        total_count += masked_patch_data.count(axis=(0, 2, 3))

    # Calculate mean and std
    # Avoid division by zero for channels with no valid data
    mean = np.divide(total_sum, total_count, out=np.zeros_like(total_sum), where=total_count != 0)
    
    # E[X^2] - (E[X])^2
    variance = np.divide(total_sum_sq, total_count, out=np.zeros_like(total_sum_sq), where=total_count != 0) - np.square(mean)
    std = np.sqrt(variance)

    print("\nMean for each channel:")
    print(mean)
    print("\nStandard deviation for each channel:")
    print(std)

if __name__ == "__main__":
    compute_mean_std()
