"""
In this module we will use helper to create a proper diffusion setupcre
"""

import torch

import numpy as np
import scipy.fft

import torch_dct as dct
import einops


from meteolibre_model.diffusion.utils import MEAN_CHANNEL, STD_CHANNEL

# -- Parameters --
BLUR_SIGMA_MAX = 1.0

import torch
import numpy as np
from typing import Tuple
import torch_dct as dct

# -- Parameters --
BLUR_SIGMA_MAX = 1.0
D_MIN = 0.001


def get_blurring_diffusion_schedules_pt(
    t_batch: torch.Tensor,
    height: int,
    width: int,
    sigma_b_max: float,
    d_min: float = D_MIN,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes combined blurring and noise schedules for a batch of timesteps `t`
    using PyTorch operations on the target device.

    Args:
        t_batch (torch.Tensor): A 1D tensor of timesteps (between 0 and 1) for the batch.
        height (int): Image height.
        width (int): Image width.
        sigma_b_max (float): Maximum blur sigma.
        d_min (float): Minimum damping factor for high frequencies.

    Returns:
        A tuple containing:
        - alpha_t_vec (torch.Tensor): The frequency-dependent signal scaling tensor
          of shape (B, 1, H, W).
        - sigma_t_batch (torch.Tensor): The scalar noise standard deviation tensor
          of shape (B,).
    """
    device = t_batch.device
    batch_size = t_batch.size(0)

    # 1. Standard Gaussian Noise Schedule (Cosine Schedule)
    # Reshape t for broadcasting over H and W dimensions
    t = t_batch.view(-1, 1, 1)
    a_t_batch = torch.cos(t * torch.pi / 2.0)
    sigma_t_batch = torch.sin(t_batch * torch.pi / 2.0)  # Keep as (B,)

    # 2. Blurring Schedule (sin^2 schedule for blur amount)
    sigma_b_t = sigma_b_max * (torch.sin(t_batch * torch.pi / 2.0) ** 2)

    # 3. Calculate the frequency-dependent damping factor d_t
    dissipation_time = (sigma_b_t**2) / 2.0  # Shape: (B,)

    # Calculate frequencies (lambda) on the device
    freqs_h = torch.linspace(0, height - 1, height, device=device)
    freqs_w = torch.linspace(0, width - 1, width, device=device)
    lambda_h = (np.pi * freqs_h / height) ** 2
    lambda_w = (np.pi * freqs_w / width) ** 2

    # Create a 2D grid of lambda values
    lambda_2d = lambda_h[:, None] + lambda_w[None, :]  # Shape: (H, W)

    # Reshape dissipation time for broadcasting: (B,) -> (B, 1, 1)
    dissipation_time_b = dissipation_time.view(batch_size, 1, 1)

    # d_t = (1 - d_min) * exp(-lambda * tau_t) + d_min
    # Unsqueeze lambda_2d to (1, H, W) to broadcast with dissipation_time_b (B, 1, 1)
    # The result will broadcast to (B, H, W)
    d_t_vec = (1 - d_min) * torch.exp(
        -lambda_2d.unsqueeze(0) * dissipation_time_b
    ) + d_min

    # 4. Combine schedules: alpha_t = a_t * d_t
    # a_t_batch is (B, 1, 1) and d_t_vec is (B, H, W), which broadcasts to (B, H, W)
    alpha_t_vec = a_t_batch * d_t_vec

    # Add the channel dimension for broadcasting with the image tensor (B, C, H, W)
    alpha_t_vec = alpha_t_vec.unsqueeze(1)

    return alpha_t_vec, sigma_t_batch


def apply_blur_diffusion(
    image_batch: torch.Tensor,
    t_batch: torch.Tensor,
    sigma_b_max: float = BLUR_SIGMA_MAX,
) -> torch.Tensor:
    """
    Applies a frequency-based blur to a batch of images based on a diffusion
    timestep `t`.

    This function performs the operation for the entire batch in a vectorized
    manner on the specified device (e.g., GPU), avoiding slow loops and
    CPU-GPU data transfers.

    Args:
        image_batch (torch.Tensor): The input images, expected to be a tensor of shape
                                    (B, C, H, W), where B is the batch size,
                                    C is the number of channels.
        t_batch (torch.Tensor): A 1D tensor of timesteps, with one value per image
                                in the batch. Shape: (B,). Values should be
                                between 0 (no blur) and 1 (max blur).
        sigma_b_max (float): The maximum blur sigma, controlling the blur intensity at t=1.

    Returns:
        torch.Tensor: The batch of blurred images with the same shape as the input.
    """
    # --- 1. Input Validation and Setup ---
    if image_batch.dim() != 4:
        raise ValueError(
            f"Expected image_batch to be a 4D tensor, but got {image_batch.dim()} dimensions."
        )
    if t_batch.dim() != 1 or t_batch.shape[0] != image_batch.shape[0]:
        raise ValueError(
            "t_batch must be a 1D tensor with the same length as the batch size of image_batch."
        )

    device = image_batch.device
    b, c, h, w = image_batch.shape
    t_batch = t_batch.to(device)

    # --- 2. Get Blurring Schedules ---
    # We only need alpha_t for blurring, so we ignore sigma_t.
    # The shape of alpha_t_batch will be (B, 1, H, W).
    alpha_t_batch, sigma_t_batch = get_blurring_diffusion_schedules_pt(
        t_batch, h, w, sigma_b_max
    )

    # --- 3. Transform to Frequency Domain ---
    # torch.fft.rfft2 is efficient for real-valued inputs like images.
    image_freq = dct.dct_2d(image_batch)

    # --- 4. Apply Blurring in Frequency Domain ---
    # The output of rfft2 has a different size on the last dimension (w // 2 + 1).
    # We must slice alpha_t to match it before multiplication.
    alpha_t_batch_sliced = alpha_t_batch[..., : image_freq.shape[-1]]

    blurred_image_freq = alpha_t_batch_sliced * image_freq

    # --- 5. Transform Back to Pixel Domain ---
    # We provide the original dimensions `s=(h, w)` to ensure the output is correctly sized.
    noise = torch.randn_like(image_batch)

    blurred_image = dct.idct_2d(blurred_image_freq)

    return blurred_image.detach() + noise * sigma_t_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).detach()

# --- Helper function to add standard isotropic noise (for comparison) ---


def trainer_step(model, batch, device):
    """
    Performs a single training step for a blurring diffusion model.

    Args:
        model: The neural network model. It is expected to take a tensor of shape
               (BATCH, NB_CHANNEL, 6, H, W) and a timestep `t` as input, and
               return the predicted noise of shape (BATCH, NB_CHANNEL, 2, H, W).
        batch_data: A tensor of shape (BATCH, 6, NB_CHANNEL, H, W), where the
                    first 4 time steps are the context and the last 2 are the target.

    Returns:
        The loss value for the training step.
    """
    with torch.no_grad():
        # The model expects (BATCH, NB_CHANNEL, NB_TEMPORAL, H, W), so permute dimensions
        batch_data = batch["patch_data"].permute(0, 2, 1, 3, 4)

        b, c, t, h, w = batch_data.shape

        # normalize using MEAN and STD
        batch_data = (
            (
                batch_data
                - MEAN_CHANNEL.unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .to(device)
            )
            / STD_CHANNEL.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
            * 4.0
        )

        # garde fou
        batch_data = batch_data.clamp(-8, 8)

        x_context = batch_data[:, :, :4]  # Shape: (BATCH, NB_CHANNEL, 4, H, W)
        x_target = batch_data[:, :, 4:]  # This is x_0, shape: (BATCH, NB_CHANNEL, 2, H, W)

        # 1. Generate random timesteps for the batch
        t_batch = torch.rand(batch_data.size(0), device=batch_data.device)

        # 2. Generate noise
        noise = torch.randn_like(x_target)

        # 3. Create noisy target by applying blurring diffusion forward process
        # quick permutation to manage the 4D constraint
        t_subset = 2
        x_target = einops.rearrange(x_target, "b c t h w -> (b t) c h w")

        t_blur = t_batch.unsqueeze(1).repeat(1, t_subset)
        t_blur = einops.rearrange(t_blur, "b t-> (b t)")

        x_t_batch = apply_blur_diffusion(x_target, t_blur).detach()

        x_t_batch = einops.rearrange(x_t_batch, "(b t) c h w -> b c t h w", b=b, t=t_subset)


    # 4. Get model prediction
    # The model input is the concatenation of the context and the noisy target.
    model_input = torch.cat(
        [x_context, x_t_batch], dim=2
    )  # Shape: (BATCH, NB_CHANNEL, 6, H, W)

    context_info = batch["spatial_position"]
    context_global = torch.cat([context_info, t_batch.unsqueeze(1)], dim=1)

    # The model predicts the noise `epsilon` based on the noisy image `x_t` and timestep `t`.
    predicted_noise = model(model_input.float(), context_global.float())

    # 5. Apply loss function (MSE between actual and predicted noise)
    loss = torch.nn.functional.mse_loss(predicted_noise, noise.float())

    return loss


def full_image_generation(model, batch, x_context, steps=1000, device="cuda"):
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
        x_context = x_context[[0]]  # batch of size 1 to reduce compute

        context_info = batch["spatial_position"].to(device)

        batch_size, nb_channel, _, h, w = x_context.shape
        delta = 1e-8  # Small constant for numerical stability [cite: 472]
        delta_tensor = torch.tensor(delta, device=device)

        # 1. Start with pure Gaussian noise (z_T) [cite: 204]
        z_t = torch.randn(batch_size, nb_channel, 2, h, w, device=device)

        # 2. Loop from T to 1 [cite: 205]
        for i in range(steps, 0, -1):
            t_val = i / steps
            s_val = (i - 1) / steps

            # Create a tensor for the current timestep
            t_batch = torch.full(
                (batch_size,), t_val, device=device, dtype=torch.float32
            )
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
            hat_eps_t_reshaped = einops.rearrange(
                hat_eps_t, "b c t h w -> (b t) c h w"
            )

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
            alpha_ts = alpha_t / (alpha_s + delta_tensor)
            sigma2_ts = sigma_t**2 - alpha_ts**2 * sigma_s**2

            # Calculate denoising variance (posterior variance) [cite: 179]
            sigma2_denoise = 1.0 / torch.maximum(
                (1.0 / torch.maximum(sigma_s**2, delta_tensor))
                + (alpha_ts**2 / torch.maximum(sigma2_ts, delta_tensor)),
                delta_tensor,
            )

            # Calculate denoising mean (hat_mu) [cite: 216]
            # This is a direct implementation of Equation 23
            coeff1 = (
                sigma2_denoise * alpha_ts / torch.maximum(sigma2_ts, delta_tensor)
            )
            coeff2 = (
                sigma2_denoise
                / (alpha_s * torch.maximum(sigma_s**2, delta_tensor))
            )

            term_in_eps = u_t - sigma_t * hat_u_eps_t
            hat_mu_t_s = coeff1 * u_t + coeff2 * term_in_eps

            # --- Sample z_s from the denoising distribution ---
            if i > 1:
                # Generate random noise for the sampling step
                noise_z = torch.randn_like(z_t)
                noise_z_reshaped = einops.rearrange(
                    noise_z, "b c t h w -> (b t) c h w"
                )
                u_noise_reshaped = dct.dct_2d(noise_z_reshaped, norm="ortho")
                u_noise = einops.rearrange(
                    u_noise_reshaped, "(b t) c h w -> b c t h w", b=batch_size
                )

                # Sample u_s = mean + std * noise [cite: 207]
                u_s = hat_mu_t_s + torch.sqrt(sigma2_denoise) * u_noise
            else:
                # The last step is deterministic
                u_s = hat_mu_t_s

            # Convert back to pixel space and update z_t for the next iteration
            u_s_reshaped = einops.rearrange(u_s, "b c t h w -> (b t) c h w")
            z_s_reshaped = dct.idct_2d(u_s_reshaped, norm="ortho")
            z_t = einops.rearrange(
                z_s_reshaped, "(b t) c h w -> b c t h w", b=batch_size
            )

    # Set model back to training mode
    model.train()

    # The final z_0 is our generated sample
    return z_t.cpu()
