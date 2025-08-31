"""
Rectified Flow implementation for weather forecasting diffusion model.
This module provides functions for training and generation using rectified flow.
"""

import torch
import einops

from meteolibre_model.diffusion.utils import MEAN_CHANNEL, STD_CHANNEL

# -- Parameters --
CLIP_MIN = -4

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

def get_x_t(x0, x1, t):
    """
    Get the interpolated point x_t = (1 - t^{1/4}) * x0 + t^{1/4} * x1
    """
    t_p = t ** (1/4)
    return (1 - t_p) * x0 + t_p * x1

def trainer_step(model, batch, device):
    """
    Performs a single training step for the rectified flow model.

    Args:
        model: The neural network model.
        batch: Batch data from the dataset.
        device: Device to run on.

    Returns:
        The loss value for the training step.
    """
    with torch.no_grad():
        # Permute to (B, C, T, H, W)
        batch_data = batch["patch_data"].permute(0, 2, 1, 3, 4)

        b, c, t, h, w = batch_data.shape

        # Normalize
        batch_data = normalize(batch_data, device)

        x_context = batch_data[:, :, :4]  # Context frames
        x_target = batch_data[:, :, 4:]  # Target frames (x0)

        mask_data = x_target != CLIP_MIN

        # Sample timesteps
        t_batch = torch.rand(b, device=device)

        # Generate noise (x1)
        x1 = torch.randn_like(x_target)

        # Expand t for broadcasting: (b,) -> (b, 1, 2, 1, 1)
        t_expanded = t_batch.view(b, 1, 1, 1, 1).expand(-1, -1, 2, -1, -1)

        # Get x_t
        x_t = get_x_t(x_target, x1, t_expanded)

        # True velocity v = (1/4) * t^{-3/4} * (x1 - x0)
        t_expanded_full = t_batch.view(b, 1, 1, 1, 1).expand(b, c, 2, h, w)
        t_clamped = torch.clamp(t_expanded_full, min=1e-6)
        true_v = (1/4) * (t_clamped ** (-3/4)) * (x1 - x_target)

    # Model input: concatenate context and x_t
    model_input = torch.cat([x_context, x_t], dim=2)  # (B, C, 6, H, W)

    context_info = batch["spatial_position"]
    context_global = torch.cat([context_info, t_batch.unsqueeze(1)], dim=1)

    # Predict velocity
    predicted_v = model(model_input.float(), context_global.float())

    # Loss: MSE between predicted and true velocity
    loss = torch.nn.functional.mse_loss(predicted_v[mask_data], true_v[mask_data])

    return loss

def full_image_generation(model, batch, x_context, steps=100, device="cuda"):
    """
    Generates full images using rectified flow ODE solver.

    Args:
        model: The neural network model.
        batch: Batch data for context.
        x_context: Context frames.
        steps: Number of ODE steps.
        device: Device to run on.

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

        # Start with noise (x1)
        x_t = torch.randn(batch_size, nb_channel, 2, h, w, device=device)

        dt = 1.0 / steps

        for i in range(steps):
            t_val = 1 - i * dt
            t_next_val = 1 - (i + 1) * dt
            t_batch = torch.full((batch_size,), t_val, device=device)
            t_next_batch = torch.full((batch_size,), t_next_val, device=device)

            context_global = torch.cat([context_info, t_batch.unsqueeze(1)], dim=1)

            model_input = torch.cat([x_context, x_t], dim=2)
            v1 = model(model_input.float(), context_global.float())

            # Euler prediction for Heun's
            x_euler = x_t - dt * v1

            # Predict v at next t with Euler prediction
            context_global_next = torch.cat([context_info, t_next_batch.unsqueeze(1)], dim=1)
            model_input_next = torch.cat([x_context, x_euler], dim=2)
            v2 = model(model_input_next.float(), context_global_next.float())

            # Heun's step: x_{t-dt} = x_t - (dt/2) * (v1 + v2)
            x_t = x_t - (dt / 2) * (v1 + v2)

            # Clamp to prevent divergence
            x_t = x_t.clamp(-7, 7)

    model.train()
    return x_t.cpu()