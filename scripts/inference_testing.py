"""
Module to test the different inference configuration
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration, LoggerType
from tqdm.auto import tqdm
import random
import matplotlib.pyplot as plt

from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import save_file, load_file

# Add project root to sys.path
project_root = os.path.abspath("/workspace/meteolibre_model/")
sys.path.insert(0, project_root)


from meteolibre_model.dataset.dataset import MeteoLibreMapDataset
from meteolibre_model.diffusion.blurring_diffusion import (
    trainer_step,
    full_image_generation,
    BLUR_SIGMA_MAX,
)
from meteolibre_model.models.dc_3dunet_film import UNet_DCAE_3D


def main():
    # Initialize Accelerator with bfloat16 precision and logging
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=".",
        kwargs_handlers=[kwargs],
    )
    device = accelerator.device

    MODEL_DIR = "models/"
    batch_size = 128
    learning_rate = 1e-4
    num_epochs = 200
    seed = 44

    # Set seed for reproducibility
    set_seed(seed)


    # Initialize dataset
    dataset = MeteoLibreMapDataset(
        localrepo="/workspace/data/data",  # Replace with your dataset path
        cache_size=8,
        seed=seed,
    )

    # Initialize DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # os.cpu_count() // 2,  # Use half the available CPUs
        pin_memory=True,
    )

    # Initialize model
    model = UNet_DCAE_3D(
        in_channels=12,  # Adjust based on your data
        out_channels=12,  # Adjust based on your data
        features=[32, 64, 128, 256],
        context_dim=4,
        context_frames=4,
    )

    # weight from previous run
    file_path = "/workspace/meteolibre_model/models/epoch_6.safetensors"
    loaded = load_file(file_path)
    model.load_state_dict(loaded)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare for distributed training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for batch in dataloader:
        break

    if accelerator.is_main_process:
        with torch.no_grad():
            permuted_batch_data = batch["patch_data"].permute(0, 2, 1, 3, 4)


            x_context = permuted_batch_data[:, :, :4]
            x_target = permuted_batch_data[:, :, 4:]

            x_target = normalize(x_target, device)

            unwrapped_model = accelerator.unwrap_model(model)
            generated_images = full_image_generation_custom(
                unwrapped_model, batch, x_context, device=accelerator.device
            )


            # Select one channel and one batch item for visualization
            generated_sample = generated_images[0, -1]  # Shape: (2, H, W)
            target_sample = x_target[0, -1].cpu()  # Shape: (2, H, W)

            target_sample = target_sample.clamp(-1, 10)

            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            plt.imshow(generated_sample[0])
            plt.colorbar()
            plt.savefig("generated.png")

            plt.figure(figsize=(10, 10))
            plt.imshow(target_sample[0])
            plt.colorbar()
            plt.savefig("target.png")


import einops
import torch_dct as dct

from meteolibre_model.diffusion.blurring_diffusion import get_blurring_diffusion_schedules_pt, normalize

def full_image_generation_custom(model, batch, x_context, steps=1000, device="cuda"):
    """
    Generates full images using the DDPM ancestral sampler for Blurring Diffusion.

    This function implements Algorithm 1 from the paper[cite: 203]. It starts with pure
    noise and iteratively denoises it over a series of timesteps to generate the
    final target frames based on the context frames.

    Args:
        model: The neural network model that predicts noise `epsilon`.
        x_context: The context tensor of shape (BATCH, NB_CHANNEL, 4, H, W).
        steps (int): The number of denoising steps (T in the paper).
        device: The device to perform generation on ('cuda' or 'cpu').

    Returns:
        The generated image tensor of shape (BATCH, NB_CHANNEL, 2, H, W).
    """
    model.eval()
    with torch.no_grad():
        model.to(device)

        x_context = x_context.to(device)
        x_context = x_context[[0], :, :, :, :]  # batch of size 1 to reduce compute

        x_context = normalize(x_context, device)

        context_info = batch["spatial_position"].to(device)[[0], :]

        batch_size, nb_channel, _, h, w = x_context.shape
        delta = 1e-8  # Small constant for numerical stability [cite: 472]
        delta_tensor = torch.tensor(delta, device=device)

        # 1. Start with pure Gaussian noise (z_T) [cite: 204]
        z_t = torch.randn(batch_size, nb_channel, 2, h, w, device=device)

        # 2. Loop from T to 1 [cite: 205]
        for i in range(steps - 1, 0, -1):
            t_val = i / steps
            s_val = (i - 1) / steps

            # Create a tensor for the current timestep
            t_batch = torch.full((batch_size,), t_val, device=device, dtype=torch.float32)

            s_batch = torch.full(
                (batch_size,), s_val, device=device, dtype=torch.float32
            )

            context_global = torch.cat([context_info, t_batch.unsqueeze(1)], dim=1)

            # Get the model's noise prediction
            model_input = torch.cat([x_context, z_t], dim=2)
            hat_eps_t = model(model_input.float(), context_global.float())

            # --- Prepare for denoising calculation in frequency space ---
            # Reshape for 2D DCT: (b, c, t, h, w) -> (b*t, c, h, w)
            z_t_reshaped = einops.rearrange(z_t, "b c t h w -> (b t) c h w")
            hat_eps_t_reshaped = einops.rearrange(hat_eps_t, "b c t h w -> (b t) c h w")

            # Transform to frequency space u_t = V^T * z_t [cite: 205]
            u_t_reshaped = dct.dct_2d(z_t_reshaped, norm="ortho")
            hat_u_eps_t_reshaped = dct.dct_2d(hat_eps_t_reshaped, norm="ortho")

            # Reshape back: (b*t, c, h, w) -> (b, c, t, h, w)
            u_t = einops.rearrange(
                u_t_reshaped, "(b t) c h w -> b c t h w", b=batch_size
            )
            hat_u_eps_t = einops.rearrange(
                hat_u_eps_t_reshaped, "(b t) c h w -> b c t h w", b=batch_size
            )

            # --- Calculate the parameters for the denoising distribution p(z_s | z_t) ---
            # Get schedules for current (t) and previous (s) timesteps
            alpha_t_vec, sigma_t_batch = get_blurring_diffusion_schedules_pt(
                t_batch, h, w, BLUR_SIGMA_MAX
            )
            alpha_s_vec, sigma_s_batch = get_blurring_diffusion_schedules_pt(
                s_batch, h, w, BLUR_SIGMA_MAX
            )

            # Reshape for broadcasting with 5D tensors
            # alpha_t_vec is (b, 1, h, w), needs to be (b, 1, 1, h, w)
            alpha_t = alpha_t_vec.unsqueeze(2)
            alpha_s = alpha_s_vec.unsqueeze(2)
            # sigma_t_batch is (b,), needs to be (b, 1, 1, 1, 1)
            sigma_t = sigma_t_batch.view(batch_size, 1, 1, 1, 1)
            sigma_s = sigma_s_batch.view(batch_size, 1, 1, 1, 1)

            # Calculate coefficients for the mean of the denoising distribution [cite: 216]
            alpha_ts = torch.maximum(alpha_t / (alpha_s + delta_tensor), torch.tensor(1e-3, device=device))
            sigma2_ts = sigma_t**2 - alpha_ts**2 * sigma_s**2

            # Calculate denoising variance (posterior variance) [cite: 179]
            sigma2_denoise = 1.0 / torch.maximum(
                (1.0 / torch.maximum(sigma_s**2, delta_tensor))
                + (alpha_ts**2 / torch.maximum(sigma2_ts, delta_tensor)),
                delta_tensor,
            )

            # Calculate denoising mean (hat_mu) [cite: 216]
            # This is a direct implementation of Equation 23
            coeff1 = sigma2_denoise * alpha_ts / torch.maximum(sigma2_ts, delta_tensor)
            coeff2 = sigma2_denoise / (
                alpha_ts * torch.maximum(sigma_s**2, delta_tensor)
            )

            term_in_eps = u_t - sigma_t * hat_u_eps_t
            hat_mu_t_s = coeff1 * u_t + coeff2 * term_in_eps # coeff 2 is bad here

            # --- Sample z_s from the denoising distribution ---
            if i > 1:
                # Generate random noise for the sampling step
                noise_z = torch.randn_like(z_t)
                noise_z_reshaped = einops.rearrange(noise_z, "b c t h w -> (b t) c h w")
                u_noise_reshaped = dct.dct_2d(noise_z_reshaped, norm="ortho")
                u_noise = einops.rearrange(
                    u_noise_reshaped, "(b t) c h w -> b c t h w", b=batch_size
                )

                # Sample u_s = mean + std * noise [cite: 207]
                u_s = hat_mu_t_s #+ torch.sqrt(sigma2_denoise) * u_noise
            else:
                # The last step is deterministic
                u_s = hat_mu_t_s

            # Convert back to pixel space and update z_t for the next iteration
            u_s_reshaped = einops.rearrange(u_s, "b c t h w -> (b t) c h w")
            z_s_reshaped = dct.idct_2d(u_s_reshaped, norm="ortho")
            z_t = einops.rearrange(
                z_s_reshaped, "(b t) c h w -> b c t h w", b=batch_size
            )

            z_t = z_t.clamp(-10, 10)

            #breakpoint()

            z_t_previous = z_t.clone().detach()

            #breakpoint()
            

            if i % 100 == 0:
                plt.figure(figsize=(10, 5))
                plt.imshow(z_t[0, 0, 0].cpu().numpy())
                plt.colorbar()
                plt.savefig(f"intermediate_z_t_step_{i}.png")
                plt.close()

    # Set model back to training mode
    model.train()

    # The final z_0 is our generated sample
    return z_t.cpu()



if __name__ == "__main__":
    main()
