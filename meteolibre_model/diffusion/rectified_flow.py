"""
Rectified Flow implementation for weather forecasting diffusion model.
This module provides functions for training and generation using rectified flow.
"""

import torch
import einops
import math

from meteolibre_model.diffusion.utils import MEAN_CHANNEL, STD_CHANNEL

# -- Parameters --
CLIP_MIN = -4
COEF_NOISE = 3

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def logsnr_schedule_cosine(t, logsnr_min=-15, logsnr_max=15):

    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))

def logsnr_schedule_cosine_shifted(coef, t):

    logsnr_t = logsnr_schedule_cosine(t)
    return logsnr_t + 2 * math.log(coef)

def get_proper_noise(coef, t):

    t_shift = 1. - t

    logsnr_t = logsnr_schedule_cosine_shifted(coef, t_shift)
    beta_t = torch.sqrt(torch.sigmoid(logsnr_t))

    return 1. - beta_t, logsnr_t



def get_logsnr(coef, t):
    return logsnr_schedule_cosine_shifted(coef, t).clamp(min=-15, max=15)

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
    Get the interpolated point x_t = s(t) * x0 + (1 - s(t)) * x1
    """
    s_t, logsnr_t = get_proper_noise(COEF_NOISE, t)

    return s_t * x0 + (1 - s_t) * x1, logsnr_t

def ds_dt(t):
    """
    Computes the derivative of s(t) = 1 - beta_t where beta_t is derived from a shifted cosine schedule.
    s(t) = 1 - sqrt(sigmoid(logsnr_schedule_cosine_shifted(3, 1-t)))
    """
    t_clamped = torch.clamp(t, min=1e-6, max=1-1e-6)
    t_shift = 1. - t_clamped

    # Recompute intermediate values from get_proper_noise
    logsnr_t = logsnr_schedule_cosine_shifted(COEF_NOISE, t_shift)
    sigmoid_logsnr_t = torch.sigmoid(logsnr_t)
    sqrt_sigmoid_logsnr_t = torch.sqrt(sigmoid_logsnr_t)

    # Derivative of sigmoid
    d_sigmoid = sigmoid_logsnr_t * (1 - sigmoid_logsnr_t)

    # Derivative of logsnr_schedule_cosine
    t_min = math.atan(math.exp(-0.5 * 15))
    t_max = math.atan(math.exp(-0.5 * -15))
    tan_val = torch.tan(t_min + t_shift * (t_max - t_min))
    d_logsnr = -2 * (t_max - t_min) / tan_val * (1 / torch.cos(t_min + t_shift * (t_max - t_min)))**2

    # Chain rule for ds/dt
    # ds/dt = d(1-beta_t)/dt = -d(beta_t)/dt
    # d(beta_t)/dt = d(sqrt(sigmoid))/d(sigmoid) * d(sigmoid)/d(logsnr) * d(logsnr)/d(t_shift) * d(t_shift)/dt
    # d(t_shift)/dt = -1
    d_beta_dt = (0.5 / sqrt_sigmoid_logsnr_t) * d_sigmoid * d_logsnr * (-1)

    return -d_beta_dt

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
        batch_data = batch["patch_data"].permute(0, 2, 1, 3, 4)

        b, c, t, h, w = batch_data.shape

        # Normalize
        batch_data = normalize(batch_data, device)

        x_context = batch_data[:, :, :4]  # Context frames
        # Always forecast the residual
        x_target = batch_data[:, :, 4:] - batch_data[:, :, 3:4]  # Residual

        mask_data = x_target != CLIP_MIN

        # Sample timesteps
        t_batch = torch.rand(b, device=device)

        # Generate noise (x1)
        x1 = torch.randn_like(x_target)

        # Expand t for broadcasting: (b,) -> (b, 1, 2, 1, 1)
        t_expanded = t_batch.view(b, 1, 1, 1, 1).expand(-1, -1, 2, -1, -1)

        # Get x_t
        x_t, logsnr_t = get_x_t(x_target, x1, t_expanded)

        if parametrization == "standard":
            # True velocity v = ds/dt * (x0 - x1)
            t_expanded_full = t_batch.view(b, 1, 1, 1, 1).expand(b, c, 2, h, w)
            target = ds_dt(t_expanded_full) * (x_target - x1)
        elif parametrization == "endpoint":
            target = x_target
        else:
            raise ValueError(f"Unknown parametrization: {parametrization}")


    # Model input: concatenate context and x_t
    model_input = torch.cat([x_context, x_t], dim=2)  # (B, C, 6, H, W)

    context_info = batch["spatial_position"]
    context_global = torch.cat([context_info, logsnr_t.squeeze()[:, [0]] / 10.], dim=1)

    # Predict residual
    prediction = model(model_input.float(), context_global.float())[:, :, 4:, :, :]

    # Loss: MSE between predicted and true residual
    loss = torch.nn.functional.mse_loss(prediction[mask_data], target[mask_data])

    return loss

def full_image_generation(model, batch, x_context, steps=100, device="cuda", parametrization="standard"):
    """
    Generates full images using rectified flow ODE solver.

    Args:
        model: The neural network model.
        batch: Batch data for context.
        x_context: Context frames.
        steps: Number of ODE steps.
        device: Device to run on.
        parametrization: Type of parametrization ("standard" or "endpoint").

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

            s_t, logsnr_t = get_proper_noise(COEF_NOISE, t_batch)
            t_expanded_full = t_batch.view(batch_size, 1, 1, 1, 1).expand(batch_size, nb_channel, 2, h, w)
            derivative = ds_dt(t_expanded_full)

            t_next_batch = torch.full((batch_size,), t_next_val, device=device)
            s_t, logsnr_t_next = get_proper_noise(COEF_NOISE, t_batch)

            context_global = torch.cat([context_info, logsnr_t.unsqueeze(-1)[: , [0]]/ 10. ], dim=1)

            model_input = torch.cat([x_context, x_t], dim=2)
            pred = model(model_input.float(), context_global.float())[:, :, 4:, :, :]

            if parametrization == "endpoint":
                # For endpoint, prediction is residual, velocity v = pred / t_batch
                v1 = - derivative * (pred - x_t) / (s_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1e-6)
            else:
                # For standard, prediction is velocity
                v1 = pred

            # Euler prediction for Heun's
            x_euler = x_t - dt * v1

            # Predict at next t with Euler prediction
            context_global_next = torch.cat([context_info, logsnr_t_next.unsqueeze(-1)[: , [0]]/ 10.], dim=1)
            model_input_next = torch.cat([x_context, x_euler], dim=2)
            pred_next = model(model_input_next.float(), context_global_next.float())[:, :, 4:, :, :]

            if parametrization == "endpoint":
                v2 = pred_next / t_next_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                v2 = pred_next

            # Heun's step: x_{t-dt} = x_t - (dt/2) * (v1 + v2)
            x_t = x_t - (dt / 2) * (v1 + v2)

            # Clamp to prevent divergence
            x_t = x_t.clamp(-7, 7)

    # Always add back the last context since always forecasting residual
    last_context = x_context[:, :, 3:4]  # (batch_size, nb_channel, 1, h, w)
    x_t = x_t + last_context.expand(-1, -1, 2, -1, -1)

    model.train()
    return x_t.cpu()