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
from meteolibre_model.dataset.dataset_for_diffusion import MeteoFranceDataset
from meteolibre_model.model.unet_model import Unet
from meteolibre_model.diffusion.rectified_flow_lightning_shortcut import (
    full_image_generation,
    normalize,
    denormalize,
    trainer_step,
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
    if t_dim != 7:
        raise ValueError(f"Correction training requires 7 frames, but got {t_dim}")

    # Normalize all ground truth data once for use as targets and inputs
    norm_sat_data, norm_lightning_data = normalize(sat_data_full, lightning_data_full, device)
    norm_batch_data = torch.concat([norm_sat_data, norm_lightning_data], dim=1)

    # --- 2. Autoregressive Rollout with Forecast Model (no grad) ---
    with torch.no_grad():
        # Generate prediction for frame 4
        batch_gen1 = {
            "sat_patch_data": batch["sat_patch_data"][:, :5],
            "lightning_patch_data": batch["lightning_patch_data"][:, :5],
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
            "sat_patch_data": torch.cat([context_sat_gen2, batch["sat_patch_data"][:, 5:6]], dim=1),
            "lightning_patch_data": torch.cat([context_light_gen2, batch["lightning_patch_data"][:, 5:6]], dim=1),
            "spatial_position": batch["spatial_position"][:, 1, :],
        }
        pred_5_norm, _ = full_image_generation(
            forecast_model, batch_gen2, steps=generation_steps, device=device, nb_element=b, normalize_input=True
        )
        pred_5_norm = pred_5_norm.to(device)

        # Denormalize pred_4 to use in the next generation step's input
        pred_sat_4_denorm, pred_light_4_denorm = denormalize(
            pred_5_norm[:, :c_sat], pred_5_norm[:, c_sat:], device
        )

    # --- 3. Correction Model Training Step ---
    # Construct the 5-frame input for the correction model:
    # [gt_2_norm, gt_3_norm, pred_4_norm, pred_5_norm, dummy_frame]
    context_corr_model = torch.cat([
        pred_4_norm,                 # Predicted frame 4
        pred_5_norm                  # Predicted frame 5
    ], dim=2)


    batch_gen_corr = {
            "sat_patch_data": torch.cat([context_corr_model, batch["sat_patch_data"][:, 5:6]], dim=1),
            "lightning_patch_data": torch.cat([context_corr_model, batch["lightning_patch_data"][:, 5:6]], dim=1),
            "spatial_position": batch["spatial_position"][:, 1, :],
    }
    
    return trainer_step(correction_model, batch_gen_corr, device, sigma=0., interpolation="polynomial")


def main(args):
    device = torch.device(args.device)

    # --- Dataset and DataLoader ---
    dataset = MeteoFranceDataset(
        data_path=args.data_path,
        split="train",
        sequence_length=7,
        use_lightning=True
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # --- Models ---
    # Load pre-trained forecast model
    forecast_model = Unet(
        dim=64, channels=12, lightning_channels=1, out_dim=12, dim_mults=(1, 2, 4, 8)
    ).to(device)
    if not os.path.exists(args.forecast_model_path):
        raise FileNotFoundError(f"Forecast model checkpoint not found at {args.forecast_model_path}")
    forecast_model.load_state_dict(torch.load(args.forecast_model_path, map_location=device))
    print(f"Loaded forecast model from {args.forecast_model_path}")

    # Initialize correction model with the same architecture
    total_channels = 12 + 1
    correction_model = Unet(
        dim=64, channels=12, lightning_channels=1, out_dim=12, dim_mults=(1, 2, 4, 8)
    ).to(device)
    print("Initialized new correction model.")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(correction_model.parameters(), lr=args.lr)

    # --- Training Loop ---
    print("Starting training for correction model...")
    for epoch in range(args.epochs):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            loss = regression_trainer_step(
                forecast_model,
                correction_model,
                batch,
                device,
                generation_steps=args.gen_steps
            )

            loss.backward()
            optimizer.step()

            if i % args.log_interval == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.6f}")

        # --- Save Checkpoint ---
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"correction_model_epoch_{epoch+1}.pth")
            torch.save(correction_model.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a correction model for weather forecasting.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--forecast_model_path", type=str, required=True, help="Path to the pre-trained forecast model checkpoint.")
    parser.add_argument("--save_dir", type=str, default="./correction_models", help="Directory to save model checkpoints.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--gen_steps", type=int, default=4, help="Number of steps for intermediate forecast generation.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log training loss every N batches.")
    parser.add_argument("--save_interval", type=int, default=1, help="Save model checkpoint every N epochs.")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
