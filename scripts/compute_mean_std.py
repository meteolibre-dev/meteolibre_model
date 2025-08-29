import numpy as np
from datasets import load_dataset
from tqdm import tqdm

def compute_mean_std():
    """
    Computes the mean and standard deviation of the dataset channels.
    """
    # Load the dataset
    ds = load_dataset("meteolibre-dev/mtg_europe_64", streaming=True, split="train")

    # Initialize accumulators
    num_channels = 12
    total_sum = np.zeros(num_channels, dtype=np.float64)
    total_sum_sq = np.zeros(num_channels, dtype=np.float64)
    total_count = 0
    
    # Use a smaller subset for quick estimation, or iterate over the whole dataset
    # for a more accurate result. Set num_samples to None to iterate over everything.
    num_samples = 1000 # Or None

    print(f"Calculating mean and std over {num_samples or 'all'} samples...")

    for i, example in enumerate(tqdm(ds, total=num_samples)):
        if num_samples is not None and i >= num_samples:
            break

        # Get the byte string and reshape
        patch_data = np.frombuffer(
            example["data"], dtype=np.dtype(example["dtype"])
        ).reshape(example["shape"])

        # Ensure patch_data is float64 for precision in sums
        patch_data = patch_data.astype(np.float64)

        # Update accumulators
        # The shape is (6, 12, 64, 64), so we sum over axes 0, 2, and 3
        total_sum += np.sum(patch_data, axis=(0, 2, 3))
        total_sum_sq += np.sum(np.square(patch_data), axis=(0, 2, 3))
        
        # Number of values per channel in this patch
        count_per_channel = patch_data.shape[0] * patch_data.shape[2] * patch_data.shape[3]
        total_count += count_per_channel

    # Calculate mean and std
    mean = total_sum / total_count
    # E[X^2] - (E[X])^2
    variance = (total_sum_sq / total_count) - np.square(mean)
    std = np.sqrt(variance)

    print("\nMean for each channel:")
    print(mean)
    print("\nStandard deviation for each channel:")
    print(std)

if __name__ == "__main__":
    compute_mean_std()
