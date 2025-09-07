"""
Rectified Flow implementation for weather forecasting diffusion model.
This module provides functions for training and generation using rectified flow.
"""

import torch
import math

from meteolibre_model.diffusion.utils import MEAN_CHANNEL, STD_CHANNEL

# -- Parameters --
CLIP_MIN = -4
COEF_NOISE = 2

LOGSNR_MIN = -15.0
LOGSNR_MAX =  15.0
T_MIN = math.atan(math.exp(-0.5 * LOGSNR_MAX))
T_MAX = math.atan(math.exp(-0.5 * LOGSNR_MIN))

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def logsnr_schedule_cosine(t):
    return -2 * log(torch.tan(T_MIN + t * (T_MAX - T_MIN)))


def logsnr_schedule_cosine_shifted(coef, t):
    logsnr_t = logsnr_schedule_cosine(t)
    return logsnr_t + 2 * math.log(coef)


def get_proper_noise(coef, t):
    t_shift = 1.0 - t

    logsnr_t = logsnr_schedule_cosine_shifted(coef, t_shift)
    beta_t = torch.sqrt(torch.sigmoid(logsnr_t))

    return 1.0 - beta_t, logsnr_t


def get_logsnr(coef, t):
    return logsnr_schedule_cosine_shifted(coef, t).clamp(min=-15, max=15)


def normalize(batch_data, device):
    """
    Normalize the batch data using precomputed mean and std.
    """
    batch_data = (
        batch_data
        - MEAN_CHANNEL.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    ) / STD_CHANNEL.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)

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
    # t in (0,1). We clamp for safety.
    t = torch.clamp(t, 1e-6, 1 - 1e-6)

    # s(t) = 1 - sqrt(sigmoid(logsnr(1 - t) + 2 log coef))
    t_shift = 1.0 - t
    logsnr = logsnr_schedule_cosine(t_shift) + 2 * math.log(COEF_NOISE)

    sig = torch.sigmoid(logsnr)
    sqrt_sig = torch.sqrt(sig)

    # d beta / d logsnr where beta = sqrt(sig)
    d_beta_d_logsnr = 0.5 / (sqrt_sig + 1e-12) * sig * (1.0 - sig)

    # d logsnr / d t_shift = -2 * (T_MAX - T_MIN) * (tan(u) + 1/tan(u))
    u = T_MIN + t_shift * (T_MAX - T_MIN)
    tan_u = torch.tan(u)
    tan_u = torch.clamp(tan_u, min=1e-12)  # sign will come from plus/minus below
    d_logsnr_dt_shift = -2.0 * (T_MAX - T_MIN) * (tan_u + 1.0 / tan_u)

    # t_shift = 1 - t -> d t_shift / d t = -1
    d_beta_dt = d_beta_d_logsnr * d_logsnr_dt_shift * (-1.0)
    return -d_beta_dt  # ds/dt = - d beta / dt

def generate_correlated_noise(x_target_shape, device, rho=0.95):
    B, C, T, H, W = x_target_shape
    single = (B, C, H, W)
    sigma = math.sqrt(1.0 - rho * rho)

    eps = []
    last = torch.randn(*single, device=device)
    eps.append(last)
    for _ in range(1, T):
        new = rho * last + sigma * torch.randn(*single, device=device)
        eps.append(new)
        last = new
    return torch.stack(eps, dim=2)


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
        x1 = generate_correlated_noise(x_target.shape, device)

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
    context_global = torch.cat([context_info, logsnr_t.squeeze()[:, [0]] / 10.0], dim=1)

    # Predict residual
    prediction = model(model_input.float(), context_global.float())[:, :, 4:, :, :]

    # Loss: MSE between predicted and true residual
    loss = torch.nn.functional.mse_loss(prediction[mask_data], target[mask_data])

    return loss

def full_image_generation(
    model, batch, x_context, steps=200, device="cuda", parametrization="standard"
):
    model.eval()
    with torch.no_grad():
        model.to(device)
        x_context = x_context.to(device)
        x_context = x_context[[0], :, :, :, :]
        x_context = normalize(x_context, device)

        context_info = batch["spatial_position"].to(device)[[0], :]
        B, C, _, H, W = x_context.shape

        # Start noise and keep x1_fixed around if you use endpoint parametrization
        x_t = generate_correlated_noise((B, C, 2, H, W), device)
        x1_fixed = x_t.clone()  # needed only for "endpoint" parametrization

        dt = 1.0 / steps

        for i in range(steps):
            t_val = 1 - i * dt
            t_next_val = 1 - (i + 1) * dt
            t_batch = torch.full((B,), t_val, device=device)
            t_next_batch = torch.full((B,), t_next_val, device=device)

            # Context scalar for current t
            _, logsnr_t = get_proper_noise(COEF_NOISE, t_batch)
            context_global = torch.cat(
                [context_info, logsnr_t.unsqueeze(-1) / 10.0], dim=1
            )

            # Predict at current x_t
            model_input = torch.cat([x_context, x_t], dim=2)
            pred = model(model_input.float(), context_global.float())[:, :, 4:, :, :]

            # standard: model predicts velocity directly
            v1 = pred

            x_euler = x_t - dt * v1

            # Predict at next t with Euler prediction
            _, logsnr_t_next = get_proper_noise(COEF_NOISE, t_next_batch)
            context_global_next = torch.cat(
                [context_info, logsnr_t_next.unsqueeze(-1) / 10.0], dim=1
            )
            model_input_next = torch.cat([x_context, x_euler], dim=2)
            pred_next = model(model_input_next.float(), context_global_next.float())[:, :, 4:, :, :]

            if parametrization == "endpoint":
                t_next_expanded_full = t_next_batch.view(B, 1, 1, 1, 1).expand(B, C, 2, H, W)
                v2 = ds_dt(t_next_expanded_full) * (pred_next - x1_fixed)
            else:
                v2 = pred_next

            # Heun
            x_t = x_t - 0.5 * dt * (v1 + v2)

            # Optional: defer/relax clamping (see point 6)
            x_t = x_t.clamp(-7, 7)

        # add back last context (residual forecasting)
        last_context = x_context[:, :, 3:4]
        x_t = x_t + last_context.expand(-1, -1, 2, -1, -1)

    model.train()
    return x_t.cpu()
