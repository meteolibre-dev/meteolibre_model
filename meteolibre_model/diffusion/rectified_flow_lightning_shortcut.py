"""
Shortcut Rectified Flow implementation for weather forecasting diffusion model.
This module provides functions for training and generation using shortcut models.
https://arxiv.org/pdf/2410.12557
This script supports multiple interpolation schedules:
- 'linear': Standard Rectified Flow interpolation.
- 'polynomial': A cubic noise schedule inspired by https://arxiv.org/abs/2301.11093
"""

import torch
import math
import random
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # If not available, set to None

from meteolibre_model.diffusion.utils import (
    MEAN_CHANNEL,
    STD_CHANNEL,
    MEAN_LIGHTNING,
    STD_LIGHTNING,
)

from meteolibre_model.diffusion.utils import (
    MEAN_CHANNEL,
    STD_CHANNEL,
    MEAN_CHANNEL_WORLD,
    STD_CHANNEL_WORLD,
    MEAN_LIGHTNING,
    STD_LIGHTNING,
)

# -- Parameters --
CLIP_MIN = -4
SHORTCUT_M = 128  # Number of base steps (M=128 as in the paper)
SHORTCUT_K = 0.25  # Fraction of batch for self-consistency (k=1/4 as in the paper)


def normalize(sat_data, lightning_data, device):
    """
    Normalize the batch data using precomputed mean and std.
    """
    sat_data = (
        sat_data
        - MEAN_CHANNEL_WORLD.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    ) / STD_CHANNEL_WORLD.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)

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
        sat_data.to(device)
        * STD_CHANNEL_WORLD.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
        + MEAN_CHANNEL_WORLD.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    )

    lightning_data = (
        lightning_data.to(device)
        * STD_LIGHTNING.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
        + MEAN_LIGHTNING.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    )

    return sat_data, lightning_data


def get_x_t_rf(x0, x1, t, interpolation="linear"):
    """
    Get the interpolated point x_t based on the chosen schedule.
    - 'linear': x_t = (1 - t) * x0 + t * x1
    - 'polynomial': x_t = (1 - t)^3 * x0 + (1 - (1 - t)^3) * x1
    """
    if interpolation == "linear":
        return (1 - t) * x0 + t * x1
    elif interpolation == "polynomial":
        u = 1 - t
        alpha = u ** 3
        return alpha * x0 + (1 - alpha) * x1
    else:
        raise ValueError(f"Unknown interpolation schedule: {interpolation}")


def trainer_step(model, batch, device, sigma=0.0, parametrization="standard", interpolation="linear"):
    """
    Performs a single training step for the shortcut rectified flow model.

    Args:
        model: The neural network model (must now accept conditioning on t and d).
        batch: Batch data from the dataset.
        device: Device to run on.
        sigma: Noise level to add to input context (default 0.0).
        parametrization: Type of parametrization ("standard" or "endpoint"). Note: Only "standard" is adapted here.
        interpolation: Interpolation schedule ('linear' or 'polynomial').

    Returns:
        The loss value for the training step, loss_sat, loss_lightning.
    """
    if parametrization != "standard":
        raise ValueError("Shortcut adaptation currently assumes 'standard' parametrization.")

    # Permute to (B, C, T, H, W)
    sat_data = batch["sat_patch_data"].permute(0, 2, 1, 3, 4)
    lightning_data = batch["lightning_patch_data"].permute(0, 2, 1, 3, 4)

    b, c_sat, t_dim, h, w = sat_data.shape
    _, c_lightning, _, _, _ = lightning_data.shape

    mask_data_sat = sat_data != CLIP_MIN

    # Normalize
    sat_data, lightning_data = normalize(sat_data, lightning_data, device)

    batch_data = torch.concat([sat_data, lightning_data], dim=1)

    x_context = batch_data[:, :, :4]  # Context frames
    if sigma > 0:
        x_context += torch.randn_like(x_context) * sigma

    # Always forecast the residual
    x0 = batch_data[:, :, 4:] - batch_data[:, :, 3:4]  # Residual (data)

    context_info = batch["spatial_position"]

    # Sample noise (x1) for all
    x1 = torch.randn_like(x0)

    # Determine number for self-consistency
    num_self = int(b * SHORTCUT_K + 0.5)  # Round to nearest int
    num_emp = b - num_self

    loss_sat = 0.0
    loss_lightning = 0.0

    if num_emp > 0:
        # Empirical (flow-matching, d=0)
        x_context_emp = x_context[:num_emp]
        x0_emp = x0[:num_emp]
        x1_emp = x1[:num_emp]
        context_info_emp = context_info[:num_emp]
        mask_emp = mask_data_sat[:num_emp, :, 4:]

        t_emp = torch.rand(num_emp, device=device)
        d_emp = torch.zeros(num_emp, device=device)

        t_exp_emp = t_emp.view(num_emp, 1, 1, 1, 1)
        xt_emp = get_x_t_rf(x0_emp, x1_emp, t_exp_emp, interpolation)

        if interpolation == "linear":
            target_emp = x1_emp - x0_emp
        elif interpolation == "polynomial":
            u_emp = 1 - t_emp
            u_exp_emp = u_emp.view(num_emp, 1, 1, 1, 1)
            target_emp = 3 * (u_exp_emp ** 2) * (x1_emp - x0_emp)
        else:
            raise ValueError(f"Unknown interpolation schedule: {interpolation}")


        # Model input: concatenate context and x_t
        model_input_emp = torch.cat([x_context_emp, xt_emp], dim=2)  # (num_emp, C, 5, H, W)

        context_global_emp = torch.cat(
            [context_info_emp, t_emp.unsqueeze(1), d_emp.unsqueeze(1)], dim=1
        )

        model_input_sat_emp = model_input_emp[:, :c_sat]
        model_input_lightning_emp = model_input_emp[:, c_sat : (c_sat + c_lightning)]

        target_sat_emp = target_emp[:, :c_sat]
        target_lightning_emp = target_emp[:, c_sat : (c_sat + c_lightning)]

        # Predict
        sat_pred_emp, lightning_pred_emp = model(
            model_input_sat_emp.float(), model_input_lightning_emp.float(), context_global_emp.float()
        )

        # Loss: MSE between predicted and target
        loss_sat_emp = torch.nn.functional.mse_loss(
            sat_pred_emp[:, :, 4:][mask_emp],
            target_sat_emp[mask_emp].float(),
        )

        loss_lightning_emp = torch.nn.functional.mse_loss(
            lightning_pred_emp[:, :, 4:],
            target_lightning_emp.float(),
        )

        loss_sat += loss_sat_emp
        loss_lightning += loss_lightning_emp

    if num_self > 0:
        # Self-consistency (d > 0)
        x_context_self = x_context[num_emp:]
        x0_self = x0[num_emp:]
        x1_self = x1[num_emp:]
        context_info_self = context_info[num_emp:]
        mask_self = mask_data_sat[num_emp:, :, 4:]

        # Sample levels uniformly
        levels = list(range(7))  # 0 to 6 for 2^0/128 to 2^6/128=0.5
        min_d = 1.0 / SHORTCUT_M

        d_self = torch.zeros(num_self, device=device)
        t_self = torch.zeros(num_self, device=device)

        for ii in range(num_self):
            l = random.choice(levels)
            dd = (2 ** l) / SHORTCUT_M
            max_k = math.floor(1.0 / dd)
            k = random.randint(2, max_k)
            tt = k * dd
            d_self[ii] = dd
            t_self[ii] = tt

        t_exp_self = t_self.view(num_self, 1, 1, 1, 1)
        xt_self = get_x_t_rf(x0_self, x1_self, t_exp_self, interpolation)

        # Compute st: s_theta(xt, t, d)
        context_global_d = torch.cat(
            [context_info_self, t_self.unsqueeze(1), d_self.unsqueeze(1)], dim=1
        )

        model_input_d = torch.cat([x_context_self, xt_self], dim=2)

        model_input_sat_d = model_input_d[:, :c_sat]
        model_input_lightning_d = model_input_d[:, c_sat : (c_sat + c_lightning)]

        sat_st, lightning_st = model(
            model_input_sat_d.float(), model_input_lightning_d.float(), context_global_d.float()
        )

        st = torch.cat([sat_st, lightning_st], dim=1)[:, :, 4:]

        # x_mid = xt - st * d (direction towards data, decreasing t)
        d_exp_self = d_self.view(num_self, 1, 1, 1, 1)
        x_mid = xt_self - st * d_exp_self

        # t_mid = t - d
        t_mid = t_self - d_self

        # For second step, d_mid = d, but if d == min_d, d_mid = 0
        d_mid = d_self.clone()
        d_mid[d_self == min_d] = 0.0

        # Compute st_mid: s_theta(x_mid, t_mid, d_mid)
        context_global_mid = torch.cat(
            [context_info_self, t_mid.unsqueeze(1), d_mid.unsqueeze(1)], dim=1
        )

        model_input_mid = torch.cat([x_context_self, x_mid], dim=2)

        model_input_sat_mid = model_input_mid[:, :c_sat]
        model_input_lightning_mid = model_input_mid[:, c_sat : (c_sat + c_lightning)]

        sat_st_mid, lightning_st_mid = model(
            model_input_sat_mid.float(), model_input_lightning_mid.float(), context_global_mid.float()
        )

        st_mid = torch.cat([sat_st_mid, lightning_st_mid], dim=1)[:, :, 4:]

        # starget = detach( (st + st_mid) / 2 )
        starget = ((st + st_mid) / 2).detach()

        # Now predict at 2d: s_theta(xt, t, 2d)
        d_2_self = 2 * d_self
        context_global_2d = torch.cat(
            [context_info_self, t_self.unsqueeze(1), d_2_self.unsqueeze(1)], dim=1
        )

        # Use same model_input_d (since xt unchanged)
        sat_pred_2d, lightning_pred_2d = model(
            model_input_sat_d.float(), model_input_lightning_d.float(), context_global_2d.float()
        )

        pred_2d = torch.cat([sat_pred_2d, lightning_pred_2d], dim=1)[:, :, 4:]

        # Split targets and preds for loss
        starget_sat = starget[:, :c_sat]
        starget_lightning = starget[:, c_sat : (c_sat + c_lightning)]

        pred_sat_2d_slice = pred_2d[:, :c_sat]
        pred_lightning_2d_slice = pred_2d[:, c_sat : (c_sat + c_lightning)]

        # Loss: MSE between predicted at 2d and starget
        loss_sat_self = torch.nn.functional.mse_loss(
            pred_sat_2d_slice[mask_self],
            starget_sat[mask_self].float(),
        )

        loss_lightning_self = torch.nn.functional.mse_loss(
            pred_lightning_2d_slice,
            starget_lightning.float(),
        )

        loss_sat += loss_sat_self
        loss_lightning += loss_lightning_self

    return loss_sat + 1. * loss_lightning, loss_sat, loss_lightning


def full_image_generation(
    model, batch, steps=128, device="cuda", parametrization="standard", nb_element=1
):
    """
    Generates full images using shortcut rectified flow (simple Euler sampling for flexibility in steps).

    Args:
        model: The neural network model.
        batch: Batch data for context.
        steps: Number of steps (can be small for fast inference, e.g., 1, 2, 4).
        device: Device to run on.
        parametrization: Type of parametrization ("standard" or "endpoint").

    Returns:
        Generated images.
    """
    if parametrization != "standard":
        raise ValueError("Shortcut adaptation currently assumes 'standard' parametrization.")

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
        batch_data = batch_data[0:nb_element]

        x_context = batch_data[:, :, :4]  # Context frames

        last_context = x_context[:, :, 3:4]  # (batch_size, nb_channel, 1, h, w)

        context_info = batch["spatial_position"].to(device)[0:nb_element, :]

        batch_size, nb_channel, _, h, w = x_context.shape

        # Start with noise (x1)
        x_t = torch.randn(batch_size, nb_channel, 1, h, w, device=device)

        d_const = 1.0 / steps

        t_val = 1.0

        for i in range(steps):
            t_batch = torch.full((batch_size,), t_val, device=device)
            d_batch = torch.full((batch_size,), d_const, device=device)

            # Model input: concatenate context and x_t
            model_input = torch.cat([x_context, x_t], dim=2)  # (B, C, 5, H, W)

            context_global = torch.cat(
                [context_info, t_batch.unsqueeze(1), d_batch.unsqueeze(1)], dim=1
            )

            model_input_sat = model_input[:, :c_sat]
            model_input_lightning = model_input[:, c_sat : (c_sat + c_lightning)]

            # Predict
            sat_pred, lightning_pred = model(
                model_input_sat.float(), model_input_lightning.float(), context_global.float()
            )

            pred = torch.cat([sat_pred, lightning_pred], dim=1)[:, :, 4:]

            # Update: x_t = x_t - pred * d (towards data)
            x_t = x_t - pred * d_const

            # Clamp to prevent divergence
            x_t = x_t.clamp(-7, 7)

            # Update t
            t_val -= d_const

    # Always add back the last context since always forecasting residual
    last_context = x_context[:, :, 3:4]  # (batch_size, nb_channel, 1, h, w)
    x_t = x_t + last_context.expand(-1, -1, 1, -1, -1)
    
    x_t = torch.where(
        last_context == CLIP_MIN, last_context, x_t
    )

    model.train()
    return x_t.cpu(), batch_data[:, :, 4:]
