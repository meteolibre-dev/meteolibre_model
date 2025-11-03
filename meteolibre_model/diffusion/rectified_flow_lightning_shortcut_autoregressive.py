"""
Shortcut Rectified Flow implementation for weather forecasting diffusion model - Autoregressive Training.

This module provides a training loop for a shortcut rectified flow model, specifically
designed to improve autoregressive prediction by training the model on its own outputs.

The training process is as follows:
1. Given a sequence of 7 frames.
2. The `full_image_generation` function is used to perform an inference rollout for 2 steps
   to predict the 5th and 6th frames based on the preceding frames.
3. A new 5-frame sequence is constructed: [frame_3, frame_4, pred_frame_5, pred_frame_6, gt_frame_7].
4. The standard `trainer_step` function is called with this new sequence to train the model.

This approach aims to make the model more robust to the compounding errors that can
occur during multi-step autoregressive generation by reusing existing tested components.
"""

import torch

from meteolibre_model.diffusion.rectified_flow_lightning_shortcut import (
    normalize,
    denormalize,
    full_image_generation,
    trainer_step,
)

def trainer_step_autoregressive(model, batch, device, sigma=0.0, parametrization="standard", interpolation="linear", generation_steps=4):
    """
    Performs a single autoregressive training step by orchestrating generation and training.
    """
    # Permute to (B, C, T, H, W)
    sat_data_full = batch["sat_patch_data"].permute(0, 2, 1, 3, 4)
    lightning_data_full = batch["lightning_patch_data"].permute(0, 2, 1, 3, 4)

    b, c_sat, t_dim, h, w = sat_data_full.shape
    _, c_lightning, _, _, _ = lightning_data_full.shape

    if t_dim != 7:
        raise ValueError(f"Autoregressive training requires 7 frames, but got {t_dim}")

    # --- Autoregressive rollout (no grad) ---
    with torch.no_grad():
        # Create a batch for generating the 5th frame (input is not normalized yet)
        batch_gen1 = {
            "sat_patch_data": batch["sat_patch_data"][:, :5],
            "lightning_patch_data": batch["lightning_patch_data"][:, :5],
            "spatial_position": batch["spatial_position"][:, 0, :], # Use first row for frame 5
        }
        # Generate frame 5. Output is normalized.
        pred_frame_5_norm, _ = full_image_generation(
            model, batch_gen1, steps=generation_steps, device=device, nb_element=b, normalize_input=True
        )
        pred_frame_5_norm = pred_frame_5_norm.to(device)

        # Denormalize the prediction to use it in the next unnormalized batch
        pred_sat_5_denorm, pred_light_5_denorm = denormalize(
            pred_frame_5_norm[:, :c_sat], pred_frame_5_norm[:, c_sat:], device
        )

        # Create a batch for generating the 6th frame
        # Context: [frame_2, frame_3, frame_4, pred_frame_5_denorm]
        context_sat_gen2 = torch.cat([
            batch["sat_patch_data"][:, 1:4],
            pred_sat_5_denorm.permute(0, 2, 1, 3, 4)
        ], dim=1)
        context_light_gen2 = torch.cat([
            batch["lightning_patch_data"][:, 1:4],
            pred_light_5_denorm.permute(0, 2, 1, 3, 4)
        ], dim=1)

        # Add a dummy 5th frame for the function signature
        dummy_frame_sat = torch.zeros_like(batch["sat_patch_data"][:, 0:1])
        dummy_frame_lightning = torch.zeros_like(batch["lightning_patch_data"][:, 0:1])

        batch_gen2 = {
            "sat_patch_data": torch.cat([context_sat_gen2, dummy_frame_sat], dim=1),
            "lightning_patch_data": torch.cat([context_light_gen2, dummy_frame_lightning], dim=1),
            "spatial_position": batch["spatial_position"][:, 1, :], # Use second row for frame 6
        }

        # Generate frame 6. Output is normalized.
        pred_frame_6_norm, _ = full_image_generation(
            model, batch_gen2, steps=generation_steps, device=device, nb_element=b, normalize_input=True
        )
        pred_frame_6_norm = pred_frame_6_norm.to(device)

        # Denormalize prediction 6 for the final training step
        pred_sat_6_denorm, pred_light_6_denorm = denormalize(
            pred_frame_6_norm[:, :c_sat], pred_frame_6_norm[:, c_sat:], device
        )

    # --- Training on frame 7 ---
    # Construct the final 5-frame sequence for training with unnormalized data:
    # [frame_3, frame_4, pred_frame_5_denorm, pred_frame_6_denorm, gt_frame_7]
    sat_train_seq = torch.cat([
        batch["sat_patch_data"][:, 2:4],             # Unnormalized GT frames 3, 4
        pred_sat_5_denorm.permute(0, 2, 1, 3, 4),    # Denormalized predicted frame 5
        pred_sat_6_denorm.permute(0, 2, 1, 3, 4),    # Denormalized predicted frame 6
        batch["sat_patch_data"][:, 6:7]              # Unnormalized GT frame 7
    ], dim=1)

    lightning_train_seq = torch.cat([
        batch["lightning_patch_data"][:, 2:4],
        pred_light_5_denorm.permute(0, 2, 1, 3, 4),
        pred_light_6_denorm.permute(0, 2, 1, 3, 4),
        batch["lightning_patch_data"][:, 6:7]
    ], dim=1)

    batch_train = {
        "sat_patch_data": sat_train_seq,
        "lightning_patch_data": lightning_train_seq,
        "spatial_position": batch["spatial_position"][:, 2, :], # Use third row for frame 7
    }

    # Call the standard trainer_step, which will handle normalization internally
    return trainer_step(model, batch_train, device, sigma, parametrization, interpolation)