"""
Training script for a correction model in a weather forecasting pipeline.

This script implements the training of a correction model that refines the
predictions of a primary forecasting model, following a similar structure to
the autoregressive training script.

The training process is as follows:
1.  Load a batch of 7 consecutive ground truth frames (0-6).
2.  Use the pre-trained `forecast_model` in evaluation mode to generate two
    autoregressive predictions:
    - `pred_4`: Predicted from ground truth frames 0, 1, 2, 3.
    - `pred_5`: Predicted from ground truth frames 1, 2, 3 and `pred_4`.
3.  A `correction_model` (with the same architecture as the forecast model) is
    trained to predict frame 6.
4.  The input to the `correction_model` is a 5-frame sequence constructed as
    `[gt_2, gt_3, pred_4, pred_5, dummy_frame]`.
5.  The model's output for the final frame is compared directly against the
    ground truth of frame 6 (`gt_6`).
6.  The loss is a simple Mean Squared Error, and the `correction_model`'s
    weights are updated.
"""

import torch
import argparse
import os
from torch.utils.data import DataLoader

# --- Local Imports ---
from meteolibre_model.diffusion.rectified_flow_lightning_shortcut import (
    full_image_generation,
    normalize,
    denormalize,
    trainer_step,
    CLIP_MIN,
)

def regression_trainer_step(forecast_model, correction_model, batch, device, generation_steps):
    """
    Performs a single training step for the correction model using regression.
    """
    forecast_model.eval()
    correction_model.train()

    # --- 1. Data Preparation ---
    # Permute from (B, T, C, H, W) to (B, C, T, H, W)
    sat_data_full = batch["sat_patch_data"].permute(0, 2, 1, 3, 4).to(device)
    lightning_data_full = batch["lightning_patch_data"].permute(0, 2, 1, 3, 4).to(device)

    b, c_sat, t_dim, h, w = sat_data_full.shape
    if t_dim != 6:
        raise ValueError(f"Correction training requires 6 frames, but got {t_dim}")

    # --- 2. Autoregressive Rollout with Forecast Model (no grad) ---
    with torch.no_grad():
        # Generate prediction for frame 4
        batch_gen1 = {
            "sat_patch_data": batch["sat_patch_data"][:, :4],
            "lightning_patch_data": batch["lightning_patch_data"][:, :4],
            "spatial_position": batch["spatial_position"][:, 0, :],
        }
        pred_4_norm, _ = full_image_generation(
            forecast_model, batch_gen1, steps=generation_steps, device=device, nb_element=b, normalize_input=True
        )
        pred_4_norm = pred_4_norm.to(device)

        # Denormalize pred_4 to use in the next generation step's input
        pred_sat_4_denorm, pred_light_4_denorm = denormalize(
            pred_4_norm[:, :c_sat], pred_4_norm[:, c_sat:], device
        )

        # Generate prediction for frame 5
        context_sat_gen2 = torch.cat([batch["sat_patch_data"][:, 1:4], pred_sat_4_denorm.permute(0, 2, 1, 3, 4)], dim=1)
        context_light_gen2 = torch.cat([batch["lightning_patch_data"][:, 1:4], pred_light_4_denorm.permute(0, 2, 1, 3, 4)], dim=1)
        batch_gen2 = {
            "sat_patch_data": context_sat_gen2,
            "lightning_patch_data": context_light_gen2,
            "spatial_position": batch["spatial_position"][:, 1, :],
        }

        pred_5_norm, _ = full_image_generation(
            forecast_model, batch_gen2, steps=generation_steps, device=device, nb_element=b, normalize_input=True
        )
        pred_5_norm = pred_5_norm.to(device)

        # Denormalize pred_5 to use if needed
        pred_sat_5_denorm, pred_light_5_denorm = denormalize(
            pred_5_norm[:, :c_sat], pred_5_norm[:, c_sat:], device
        )

    # --- 3. Correction Model Training Step ---
    # Construct the input for the correction model: [pred_4, pred_5, gt_5] all raw
    context_sat_raw = torch.cat([pred_sat_4_denorm.permute(0, 2, 1, 3, 4), pred_sat_5_denorm.permute(0, 2, 1, 3, 4)], dim=1)  # (B, 2, C_sat, H, W) raw
    context_light_raw = torch.cat([pred_light_4_denorm.permute(0, 2, 1, 3, 4), pred_light_5_denorm.permute(0, 2, 1, 3, 4)], dim=1)  # (B, 2, C_light, H, W) raw
    dummy_sat_raw = batch["sat_patch_data"][:, 5:6]  # (B, 1, C_sat, H, W) raw
    dummy_light_raw = batch["lightning_patch_data"][:, 5:6]  # (B, 1, C_light, H, W) raw

    batch_gen_corr = {
        "sat_patch_data": torch.cat([context_sat_raw, dummy_sat_raw], dim=1),  # (B, 3, C_sat, H, W) raw
        "lightning_patch_data": torch.cat([context_light_raw, dummy_light_raw], dim=1),  # (B, 3, C_light, H, W) raw
        "spatial_position": batch["spatial_position"][:, 1, :],  # Position for frame 5
    }

    return trainer_step(correction_model, batch_gen_corr, device, sigma=0., interpolation="polynomial")


def full_correction_generation(
    model,
    batch,
    steps=128,
    device="cuda",
    parametrization="standard",
    nb_element=1,
    normalize_input=True,
):
    """
    Generates corrected images using shortcut rectified flow for the correction model (context_frames=2).

    Args:
        model: The correction neural network model.
        batch: Batch data for context (T=3: frames for positions n-1, n, dummy).
        steps: Number of steps (can be small for fast inference, e.g., 1, 2, 4).
        device: Device to run on.
        parametrization: Type of parametrization ("standard" or "endpoint").
        nb_element: Number of elements to generate.
        normalize_input: If True, normalize the input data.

    Returns:
        Generated corrected images (normalized), and original target (normalized).
    """
    if parametrization != "standard":
        raise ValueError(
            "Shortcut adaptation currently assumes 'standard' parametrization."
        )

    model.eval()
    with torch.no_grad():
        model.to(device)
        sat_data = batch["sat_patch_data"].permute(0, 2, 1, 3, 4)
        lightning_data = batch["lightning_patch_data"].permute(0, 2, 1, 3, 4)

        b, c_sat, t, h, w = sat_data.shape
        _, c_lightning, t, h, w = lightning_data.shape

        if t != 3:
            raise ValueError(f"Correction generation requires 3 frames input, but got {t}")

        mask_data_lightning = lightning_data != -10.0
        mask_data_sat = sat_data != CLIP_MIN

        if normalize_input:
            # Normalize
            sat_data, lightning_data = normalize(sat_data, lightning_data, device)

        lightning_data = torch.where(mask_data_lightning, lightning_data, CLIP_MIN)

        batch_data = torch.concat([sat_data, lightning_data], dim=1)
        batch_data = batch_data[0:nb_element]

        x_context = batch_data[:, :, : model.context_frames]  # Context frames (2)

        last_context = x_context[
            :, :, (model.context_frames - 1) : model.context_frames
        ]  # Last context frame

        context_info = batch["spatial_position"].to(device)[0:nb_element, :]

        batch_size, nb_channel, _, h, w = x_context.shape

        # Start with noise (x1) for the residual
        x_t = torch.randn(batch_size, nb_channel, 1, h, w, device=device)

        d_const = 1.0 / steps

        t_val = 1.0

        for i in range(steps):
            t_batch = torch.full((batch_size,), t_val, device=device)
            d_batch = torch.full((batch_size,), d_const, device=device)

            # Model input: concatenate context and x_t
            model_input = torch.cat([x_context, x_t], dim=2)  # (B, C, 3, H, W)

            context_global = torch.cat(
                [context_info, t_batch.unsqueeze(1), d_batch.unsqueeze(1)], dim=1
            )

            model_input_sat = model_input[:, :c_sat]
            model_input_lightning = model_input[:, c_sat : (c_sat + c_lightning)]

            # Predict
            sat_pred, lightning_pred = model(
                model_input_sat.float(),
                model_input_lightning.float(),
                context_global.float(),
            )

            pred = torch.cat([sat_pred, lightning_pred], dim=1)[
                :, :, model.context_frames :
            ]  # pred for the last position

            # Update: x_t = x_t - pred * d (towards data, since velocity points to noise? Wait, in RF, velocity is from data to noise, but in shortcut it's adapted.
            # In the code for full_image_generation, it's x_t - pred * d_const, and pred is s_theta, the velocity.
            # In rectified flow, the velocity v_theta points from x0 (data) to x1 (noise), so to sample from noise to data, we integrate -v_theta.
            # But in the code, it's subtracting pred * d, assuming pred is the velocity towards noise.
            x_t = x_t - pred * d_const

            # Clamp to prevent divergence
            x_t = x_t.clamp(-7, 7)

            # Update t
            t_val -= d_const

    # Add back the last context since forecasting residual
    x_t = x_t + last_context.expand(-1, -1, 1, -1, -1)

    # Apply mask if needed
    x_t = torch.where(last_context == CLIP_MIN, last_context, x_t)

    model.train()
    # Return the generated residual position, and the original third frame as target
    target = batch_data[:, :, model.context_frames :].cpu()
    return x_t.cpu(), target

