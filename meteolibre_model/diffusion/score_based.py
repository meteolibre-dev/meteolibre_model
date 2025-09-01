"""
Score-based generative model implementation for weather forecasting.
This module provides functions for training and generation using score-based diffusion.
"""

import torch
import einops

from meteolibre_model.diffusion.utils import MEAN_CHANNEL, STD_CHANNEL

# -- Parameters --
CLIP_MIN = -4
BETA_MIN = 0.1
BETA_MAX = 20.0

SIGMA_DATA = 1.0

def normalize(batch_data, device):
    """
    Normalize the batch data using precomputed mean and std.
    """
    batch_data = (
        (
            batch_data
            - MEAN_CHANNEL.unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .to(device)
        )
        / STD_CHANNEL.unsqueeze(0)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .to(device)
    )

    # Clamp to prevent extreme values
    batch_data = batch_data.clamp(CLIP_MIN, 4)

    return batch_data

def g(t):
    """
    Diffusion coefficient g(t) = sqrt(beta(t)), where beta(t) = beta_min + t * (beta_max - beta_min)
    """
    beta_t = BETA_MIN + t * (BETA_MAX - BETA_MIN)
    return torch.sqrt(beta_t)

def sigma_t_sq(t):
    """
    Compute sigma_t^2 = integral_0^t g(s)^2 ds = integral_0^t beta(s) ds
    Since beta(s) = beta_min + s * (beta_max - beta_min), integral = beta_min * t + (beta_max - beta_min) * t^2 / 2
    """
    return BETA_MIN * t + (BETA_MAX - BETA_MIN) * t**2 / 2

def trainer_step_edm_loss(model, batch, device, parametrization="standard"):
    """
    Performs a single training step for the score-based model using the
    optimized loss weighting and sigma sampling from Karras et al. (EDM).

    Args:
        model: The neural network model.
        batch: Batch data from the dataset.
        device: Device to run on.
        parametrization: Type of parametrization ("standard" or "residual").

    Returns:
        The loss value for the training step.
    """
    with torch.no_grad():
        # Permute to (B, C, T, H, W)
        batch_data = batch["patch_data"].permute(0, 2, 1, 3, 4)
        b, c, t_frames, h, w = batch_data.shape

        # Normalize
        batch_data = normalize(batch_data, device)

        x_context = batch_data[:, :, :4]  # Context frames
        x_target = batch_data[:, :, 4:]  # Target frames (y)

        if parametrization == "residual":
            x_target = batch_data[:, :, 4:] - batch_data[:, :, 3:4]

        mask_data = x_target != CLIP_MIN

        # --- REMOVED ---
        # Sample timesteps
        # t_batch = torch.rand(b, device=device)

        # Generate noise (n)
        noise = torch.randn_like(x_target)

        # +++ ADDED +++
        # Sample sigma from a log-normal distribution, as recommended by Karras et al.
        P_mean = -1.2
        P_std = 1.2
        log_sigma = P_mean + P_std * torch.randn(b, device=device)
        sigma_batch = torch.exp(log_sigma) # sigma has shape (b,)
        sigma_sq = sigma_batch.pow(2) # sigma^2 has shape (b,)

        # Expand for broadcasting
        # Note: We now use sigma_batch directly for noising the data
        sigma_exp = sigma_batch.view(b, 1, 1, 1, 1).expand(b, c, 2, h, w)
        sigma_sq_exp = sigma_sq.view(b, 1, 1, 1, 1).expand(b, c, 2, h, w)

        # Add noise: x_t = x_target + sigma * noise
        x_t = x_target + sigma_exp * noise

    # Model input: concatenate context and x_t
    model_input = torch.cat([x_context, x_t], dim=2)  # (B, C, 6, H, W)

    context_info = batch["spatial_position"]

    # --- REMOVED ---
    # context_global = torch.cat([context_info, t_batch.unsqueeze(1)], dim=1)

    # +++ ADDED +++
    # Condition the model on log(sigma) instead of t.
    # log(sigma) is numerically more stable and suitable for network input.
    context_global = torch.cat([context_info, log_sigma.unsqueeze(1)], dim=1)


    # Predict score
    predicted_score = model(model_input.float(), context_global.float())

    # -- New Loss Calculation based on Karras et al. 2022 --

    # 1. Calculate the loss weighting λ(σ)
    loss_weight = (sigma_sq + SIGMA_DATA**2) / (sigma_sq * SIGMA_DATA**2)
    loss_weight_exp = loss_weight.view(b, 1, 1, 1, 1) # Reshape for broadcasting

    # 2. Calculate the squared denoising error term: ||n + σ² * score_pred||²
    denoising_error_term = noise + sigma_sq_exp * predicted_score
    squared_l2_norm = denoising_error_term.pow(2)

    # 3. Apply the weight λ(σ) to the squared error.
    weighted_squared_error = loss_weight_exp * squared_l2_norm

    # 4. Compute the mean of the weighted, masked loss.
    loss = torch.mean(weighted_squared_error[mask_data])

    return loss

def full_image_generation(model, batch, x_context, steps=100, device="cuda", parametrization="standard"):
    """
    Generates full images using score-based reverse SDE solver.

    Args:
        model: The neural network model.
        batch: Batch data for context.
        x_context: Context frames.
        steps: Number of SDE steps.
        device: Device to run on.
        parametrization: Type of parametrization ("standard" or "residual").

    Returns:
        Generated images.
    """
    model.eval()
    with torch.no_grad():
        model.to(device)
        x_context = x_context.to(device)
        x_context = x_context[[0], :, :, :, :]  # Use first batch item
        x_context = normalize(x_context, device)

        context_info = batch["spatial_position"].to(device)[[0], :]

        batch_size, nb_channel, _, h, w = x_context.shape

        # Start with noise at t=1
        t_start = 1.0
        sigma_start_sq = sigma_t_sq(torch.tensor(t_start, device=device))
        sigma_start = torch.sqrt(sigma_start_sq)
        x_t = sigma_start * torch.randn(batch_size, nb_channel, 2, h, w, device=device)

        dt = 1.0 / steps

        for i in range(steps):
            t_val = 1 - i * dt
            t_batch = torch.full((batch_size,), t_val, device=device)

            context_global = torch.cat([context_info, t_batch.unsqueeze(1)], dim=1)

            model_input = torch.cat([x_context, x_t], dim=2)
            score = model(model_input.float(), context_global.float())

            # Compute g(t)
            g_val = g(t_batch)
            g_sq = g_val ** 2

            # Expand for broadcasting
            g_sq_exp = g_sq.view(batch_size, 1, 1, 1, 1).expand(batch_size, nb_channel, 2, h, w)
            g_exp = g_val.view(batch_size, 1, 1, 1, 1).expand(batch_size, nb_channel, 2, h, w)

            # Reverse SDE step: dx = g(t)^2 * score * dt + g(t) * dW
            drift = g_sq_exp * score
            diffusion_coeff = g_exp * torch.sqrt(torch.tensor(dt, device=device))
            noise = torch.randn_like(x_t)

            x_t = x_t + drift * dt + diffusion_coeff * noise

            # Clamp to prevent divergence
            x_t = x_t.clamp(-7, 7)

        if parametrization == "residual":
            last_context = x_context[:, :, 3:4]  # (batch_size, nb_channel, 1, h, w)
            x_t = x_t + last_context.expand(-1, -1, 2, -1, -1)

    model.train()
    return x_t.cpu()