"""
Script to visualize the effect of noise on satellite images at different timesteps.
"""

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import sys
import os
project_root = os.path.abspath("/workspace/meteolibre_model/")
sys.path.insert(0, project_root)

from meteolibre_model.dataset.dataset import MeteoLibreMapDataset
from meteolibre_model.diffusion.score_based import normalize, sigma_t_sq
from meteolibre_model.diffusion.utils import MEAN_CHANNEL, STD_CHANNEL

def denormalize(batch_data, device):
    """
    Denormalize the batch data using precomputed mean and std.
    """
    batch_data = (
        batch_data * STD_CHANNEL.unsqueeze(0)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .to(device) +
        MEAN_CHANNEL.unsqueeze(0)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .to(device)
    )
    return batch_data

def main():
    device = torch.device("cpu")  # Use CPU for simplicity

    # Load dataset
    dataset = MeteoLibreMapDataset(
        localrepo="/workspace/data/data",
        cache_size=1,
        seed=44,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    # Get the second batch for variety
    iterator = iter(dataloader)
    _ = next(iterator)  # Skip first
    batch = next(iterator)  # Use second

    # Process batch
    batch_data = batch["patch_data"].permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
    x_target = batch_data[:, :, 4:5]  # Take first target frame, shape (1, 12, 1, H, W)

    # Normalize
    x_target_norm = normalize(x_target, device)

    # Timesteps to visualize
    t_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Compute original (t=0)
    x_noisy_original = x_target_norm  # No noise
    x_noisy_denorm_original = denormalize(x_noisy_original, device)

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Effect of Noise on Satellite Image (Channel 0)', fontsize=16)

    for i, t in enumerate(t_values):
        t_tensor = torch.tensor(t, device=device)
        sigma_sq = sigma_t_sq(t_tensor)
        sigma = torch.sqrt(sigma_sq)

        # Generate noise
        noise = torch.randn_like(x_target_norm)

        # Add noise
        x_noisy = x_target_norm + sigma * noise

        # Denormalize
        x_noisy_denorm = denormalize(x_noisy, device)

        # Take first channel, squeeze dimensions
        img = x_noisy_denorm[0, 0, 0].cpu().numpy()

        # Plot
        ax = axes.flat[i]
        im = ax.imshow(img, cmap='viridis', vmin=img.min(), vmax=img.max())
        ax.set_title(f't={t}, σ_noise={sigma.item():.2f}, std_img={img.std():.2f}')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig('/workspace/meteolibre_model/noise_effect.png', dpi=300, bbox_inches='tight')
    print("Noise effect visualization saved to /workspace/meteolibre_model/noise_effect.png")

    # Print some statistics
    print("\nStatistics for Channel 0:")
    print(f"Original (t=0): mean={x_noisy_denorm_original[0,0,0].mean().item():.2f}, std={x_noisy_denorm_original[0,0,0].std().item():.2f}")
    for t in t_values[1:]:
        t_tensor = torch.tensor(t, device=device)
        sigma_sq = sigma_t_sq(t_tensor)
        sigma = torch.sqrt(sigma_sq)
        noise = torch.randn_like(x_target_norm)
        x_noisy = x_target_norm + sigma * noise
        x_noisy_denorm = denormalize(x_noisy, device)
        print(f"At t={t}: mean={x_noisy_denorm[0,0,0].mean().item():.2f}, std={x_noisy_denorm[0,0,0].std().item():.2f}")

if __name__ == "__main__":
    main()