"""
Rectified Flow implementation for weather forecasting diffusion model.
This module provides functions for training and generation using rectified flow.
"""

import torch
import math
import matplotlib.pyplot as plt

from meteolibre_model.diffusion.utils import (
    MEAN_CHANNEL,
    STD_CHANNEL,
    MEAN_LIGHTNING,
    STD_LIGHTNING,
)

# -- Parameters --
CLIP_MIN = -4


def normalize(sat_data, lightning_data, device):
    """
    Normalize the batch data using precomputed mean and std.
    """
    sat_data = (
        sat_data
        - MEAN_CHANNEL.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    ) / STD_CHANNEL.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)

    # Clamp to prevent extreme values
    sat_data = sat_data.clamp(CLIP_MIN, 4)

    lightning_data = (
        lightning_data
        - MEAN_LIGHTNING.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    ) / STD_LIGHTNING.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)

    # Clamp to prevent extreme values
    lightning_data = lightning_data.clamp(CLIP_MIN, 10)

    return sat_data, lightning_data


def denormalize(sat_data, lightning_data, device):
    """
    Denormalize the batch data using precomputed mean and std.
    """
    sat_data = (
        sat_data
        * STD_CHANNEL.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
        + MEAN_CHANNEL.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    )

    lightning_data = (
        lightning_data
        * STD_LIGHTNING.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
        + MEAN_LIGHTNING.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    )

    return sat_data, lightning_data


def get_x_t_rf(x0, x1, t):
    """
    Get the interpolated point x_t = t * x0 + (1 - t) * x1
    """

    return (1 - t) * x0 + t * x1

def trainer_step(model, batch, device, parametrization="standard"):
    """
    Performs a single training step for the rectified flow model.

    Args:
        model: The neural network model.
        batch: Batch data from the dataset.
        device: Device to run on.
        parametrization: Type of parametrization ("standard" or "endpoint").

    Returns:
        The loss value for the training step.
    """
    with torch.no_grad():
        # Permute to (B, C, T, H, W)
        sat_data = batch["sat_patch_data"].permute(0, 2, 1, 3, 4)
        lightning_data = batch["lightning_patch_data"].permute(0, 2, 1, 3, 4)

        b, c_sat, t, h, w = sat_data.shape
        b, c_lightning, t, h, w = lightning_data.shape

        mask_data_sat = sat_data != CLIP_MIN

        # Normalize
        sat_data, lightning_data = normalize(sat_data, lightning_data, device)

        batch_data = torch.concat([sat_data, lightning_data], dim=1)

        x_context = batch_data[:, :, :4]  # Context frames

        # Always forecast the residual
        x_target = batch_data[:, :, 4:] - batch_data[:, :, 3:4]  # Residual

        # Sample timesteps
        t_batch = torch.rand(b, device=device)

        # Generate noise (x1)
        x1 = torch.randn_like(x_target)

        # Expand t for broadcasting: (b,) -> (b, 1, 1, 1, 1)
        t_expanded = t_batch.view(b, 1, 1, 1, 1).expand(-1, -1, 1, -1, -1)

        # Get x_t
        x_t = get_x_t_rf(x_target, x1, t_expanded)

        if parametrization == "standard":
            target = x1 - x_target
        elif parametrization == "endpoint":
            target = x_target
        else:
            raise ValueError(f"Unknown parametrization: {parametrization}")

    # Model input: concatenate context and x_t
    model_input = torch.cat([x_context, x_t], dim=2)  # (B, C, 5, H, W)

    context_info = batch["spatial_position"]

    context_global = torch.cat([context_info, t_batch.unsqueeze(-1)], dim=1)

    model_input_sat = model_input[:, :c_sat]
    model_input_lightning = model_input[:, c_sat : (c_sat + c_lightning)]

    target_sat = target[:, :c_sat]
    target_lightning = target[:, c_sat : (c_sat + c_lightning)]

    # Predict
    sat_pred, lightning_pred = model(
        model_input_sat.float(), model_input_lightning.float(), context_global.float()
    )

    # Loss: MSE between predicted and target
    loss_sat = torch.nn.functional.mse_loss(
        sat_pred[:, :, 4:][mask_data_sat[:, :, 4:]],
        target_sat[mask_data_sat[:, :, 4:]].float(),
    )

    loss_lightning = torch.nn.functional.mse_loss(
        lightning_pred[:, :, 4:],
        target_lightning.float(),
    )

    return loss_sat + 1. * loss_lightning, loss_sat, loss_lightning


def full_image_generation(
    model, batch, steps=128, device="cuda", parametrization="standard"
):
    """
    Generates full images using rectified flow ODE solver.

    Args:
        model: The neural network model.
        batch: Batch data for context.
        steps: Number of ODE steps.
        device: Device to run on.
        parametrization: Type of parametrization ("standard" or "endpoint").

    Returns:
        Generated images.
    """
    model.eval()
    with torch.no_grad():
        model.to(device)
        sat_data = batch["sat_patch_data"].permute(0, 2, 1, 3, 4)
        lightning_data = batch["lightning_patch_data"].permute(0, 2, 1, 3, 4)

        b, c_sat, t, h, w = sat_data.shape
        b, c_lightning, t, h, w = lightning_data.shape

        mask_data_lightning = lightning_data != -10.0
        mask_data_sat = sat_data != CLIP_MIN

        # Normalize
        sat_data, lightning_data = normalize(sat_data, lightning_data, device)

        lightning_data = torch.where(mask_data_lightning, lightning_data, CLIP_MIN)

        batch_data = torch.concat([sat_data, lightning_data], dim=1)
        batch_data = batch_data[[0]]

        x_context = batch_data[:, :, :4]  # Context frames

        last_context = x_context[:, :, 3:4]  # (batch_size, nb_channel, 1, h, w)

        context_info = batch["spatial_position"].to(device)[[0], :]

        batch_size, nb_channel, _, h, w = x_context.shape

        # Start with noise (x1)
        x_t = torch.randn(batch_size, nb_channel, 1, h, w, device=device)

        dt = 1.0 / steps

        for i in range(steps-1):
            t_val = 1 - i * dt
            t_next_val = 1 - (i + 1) * dt
            t_batch = torch.full((batch_size,), t_val, device=device)
            t_next_batch = torch.full((batch_size,), t_next_val, device=device)

            # Avoid division by zero in endpoint mode
            t_safe = t_batch.clamp(min=1e-8)
            t_next_safe = t_next_batch.clamp(min=1e-8)

            context_global = torch.cat(
                [context_info, t_batch.unsqueeze(-1)], dim=1
            )

            model_input = torch.cat([x_context, x_t], dim=2)

            model_input_sat = model_input[:, :c_sat]
            model_input_lightning = model_input[:, c_sat : (c_sat + c_lightning)]

            sat_pred, lightning_pred = model(
                model_input_sat.float(), model_input_lightning.float(), context_global.float()
            )

            pred = torch.cat([sat_pred, lightning_pred], dim=1)[:, :, 4:]

            if parametrization == "standard":
                v1 = pred
            elif parametrization == "endpoint":
                v1 = (x_t - pred) / t_safe.view(-1, 1, 1, 1, 1)
            else:
                raise ValueError(f"Unknown parametrization: {parametrization}")

            # Euler prediction for Heun's
            x_euler = x_t - dt * v1

            # Predict at next t with Euler prediction
            context_global_next = torch.cat(
                [context_info, t_next_batch.unsqueeze(-1)], dim=1
            )
            model_input_next = torch.cat([x_context, x_euler], dim=2)

            model_input_sat_next = model_input_next[:, :c_sat]
            model_input_lightning_next = model_input_next[:, c_sat : (c_sat + c_lightning)]

            sat_pred_next, lightning_pred_next = model(
                model_input_sat_next.float(),
                model_input_lightning_next.float(),
                context_global_next.float(),
            )

            pred_next = torch.cat([sat_pred_next, lightning_pred_next], dim=1)[:, :, 4:]

            if parametrization == "standard":
                v2 = pred_next
            elif parametrization == "endpoint":
                v2 = (x_euler - pred_next) / t_next_safe.view(-1, 1, 1, 1, 1)
            else:
                raise ValueError(f"Unknown parametrization: {parametrization}")

            # Heun's step: x_{t-dt} = x_t - (dt/2) * (v1 + v2)
            x_t = x_t - (dt / 2) * (v1 + v2)

            # Clamp to prevent divergence
            x_t = x_t.clamp(-7, 7)

            # Debug: plot and save image every 10 steps
            # current_full = x_t + last_context
            # sat_gen = current_full[:, :c_sat]
            # kpi_gen = current_full[:, c_sat:]
            # sat_denorm, kpi_denorm = denormalize(sat_gen, kpi_gen, device)
            # plt.figure(figsize=(6, 6))
            # plt.imshow(sat_denorm[0, 0, 0].cpu().numpy(), cmap='viridis')
            # plt.title(f'Step {i}')
            # plt.savefig(f'debug_step_{i}.png')
            # plt.colorbar()
            # plt.close()

    # Always add back the last context since always forecasting residual
    last_context = x_context[:, :, 3:4]  # (batch_size, nb_channel, 1, h, w)
    x_t = x_t + last_context.expand(-1, -1, 1, -1, -1)

    model.train()
    return x_t.cpu(), batch_data[:, :, 4:]