import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
import sys
from datetime import datetime, timedelta
from suncalc import get_position
import pyproj

# Add project root to sys.path
project_root = os.path.abspath("/workspace/meteolibre_model/")
sys.path.insert(0, project_root)

# Constants for EPSG 27700 coordinates
X_LOW, X_HIGH = 2907000, 7039000
Y_LOW, Y_HIGH = 594000, 3134000

from meteolibre_model.models.unet3d_film import UNet3D
from meteolibre_model.diffusion.rectified_flow import (
    normalize,
    COEF_NOISE,
    get_proper_noise,
)
from safetensors.torch import load_file


def _extract_patch(image, x, y, patch_size):
    return image[..., y : y + patch_size, x : x + patch_size]


def _place_patch(full_image, patch, x, y, patch_size):
    full_image[..., y : y + patch_size, x : x + patch_size] = patch
    return full_image


@torch.no_grad()
def tiled_inference(
    model,
    initial_context,  # (B, C, 4, H_big, W_big)
    patch_size,
    steps,
    device,
    date,
    parametrization="standard",
    nb_channels=12,
    normalize_func=normalize,
    batch_size=256,
):
    model.eval()
    model.to(device)

    _, C, T_ctx, H_big, W_big = initial_context.shape

    # Transformer for coordinate conversion
    transformer = pyproj.Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

    x_t_full_res = torch.randn(1, nb_channels, 2, H_big, W_big, device=device)

    # Create two grids of patches
    # Grid 1: Standard grid
    y_starts1 = list(range(0, H_big - patch_size + 1, patch_size))
    if (H_big - patch_size) % patch_size != 0:
        y_starts1.append(H_big - patch_size)
    x_starts1 = list(range(0, W_big - patch_size + 1, patch_size))
    if (W_big - patch_size) % patch_size != 0:
        x_starts1.append(W_big - patch_size)
    patch_coords1 = [(x, y) for y in y_starts1 for x in x_starts1]

    # Grid 2: Shifted grid
    shift = patch_size // 2
    y_starts2 = list(range(shift, H_big - patch_size + 1, patch_size))
    if (H_big - patch_size - shift) % patch_size != 0 and H_big - patch_size > shift:
        y_starts2.append(H_big - patch_size)
    x_starts2 = list(range(shift, W_big - patch_size + 1, patch_size))
    if (W_big - patch_size - shift) % patch_size != 0 and W_big - patch_size > shift:
        x_starts2.append(W_big - patch_size)
    patch_coords2 = [(x, y) for y in y_starts2 for x in x_starts2]

    patch_coords = patch_coords1 + patch_coords2
    dt = 1.0 / steps

    for i in tqdm(range(steps), desc="Tiled Denoising"):
        t_val = 1 - i * dt
        t_next_val = 1 - (i + 1) * dt
        t_batch_val = torch.full((1,), t_val, device=device)
        t_next_batch_val = torch.full((1,), t_next_val, device=device)

        s_t, logsnr_t = get_proper_noise(COEF_NOISE, t_batch_val)
        s_t_next, logsnr_t_next = get_proper_noise(COEF_NOISE, t_next_batch_val)

        aggregated_velocity = torch.zeros_like(x_t_full_res, device=device)
        overlap_counts = torch.zeros_like(x_t_full_res, device=device)

        with torch.no_grad():
            for i_batch in range(0, len(patch_coords), batch_size):
                coords_batch = patch_coords[i_batch : i_batch + batch_size]

                patch_x_t_batch, patch_context_batch, context_global_batch = [], [], []

                eastings = [
                    X_LOW + (x + patch_size // 2) * 1000 for x, y in coords_batch
                ]
                northings = [
                    Y_HIGH - (y + patch_size // 2) * 1000 for x, y in coords_batch
                ]
                lons, lats = transformer.transform(eastings, northings)

                for j, (x_start, y_start) in enumerate(coords_batch):
                    patch_x_t = _extract_patch(
                        x_t_full_res, x_start, y_start, patch_size
                    )
                    patch_context = _extract_patch(
                        initial_context, x_start, y_start, patch_size
                    )

                    result = get_position(date, lons[j], lats[j])
                    spatial_position = torch.tensor(
                        [result["azimuth"], result["altitude"], lats[j] / 10.0],
                        device=device,
                    )
                    context_global = torch.cat(
                        [spatial_position.unsqueeze(0), logsnr_t.unsqueeze(-1) / 10.0],
                        dim=1,
                    )

                    patch_x_t_batch.append(patch_x_t)
                    patch_context_batch.append(patch_context)
                    context_global_batch.append(context_global)

                model_input = torch.cat(
                    [
                        torch.cat(patch_context_batch, dim=0),
                        torch.cat(patch_x_t_batch, dim=0),
                    ],
                    dim=2,
                )
                pred_batch = model(
                    model_input.float(), torch.cat(context_global_batch, dim=0).float()
                )[:, :, 4:, :, :]

                for j, (x_start, y_start) in enumerate(coords_batch):
                    aggregated_velocity[
                        ...,
                        y_start : y_start + patch_size,
                        x_start : x_start + patch_size,
                    ] += pred_batch[j : j + 1]
                    overlap_counts[
                        ...,
                        y_start : y_start + patch_size,
                        x_start : x_start + patch_size,
                    ] += 1

        overlap_counts[overlap_counts == 0] = 1
        averaged_velocity = aggregated_velocity / overlap_counts
        x_euler = x_t_full_res - dt * averaged_velocity

        aggregated_pred_next = torch.zeros_like(x_t_full_res, device=device)
        with torch.no_grad():
            for i_batch in range(0, len(patch_coords), batch_size):
                coords_batch = patch_coords[i_batch : i_batch + batch_size]

                patch_x_euler_batch, patch_context_batch, context_global_next_batch = (
                    [],
                    [],
                    [],
                )

                eastings = [
                    X_LOW + (x + patch_size // 2) * 1000 for x, y in coords_batch
                ]
                northings = [
                    Y_HIGH - (y + patch_size // 2) * 1000 for x, y in coords_batch
                ]
                lons, lats = transformer.transform(eastings, northings)

                for j, (x_start, y_start) in enumerate(coords_batch):
                    patch_x_euler = _extract_patch(
                        x_euler, x_start, y_start, patch_size
                    )
                    patch_context = _extract_patch(
                        initial_context, x_start, y_start, patch_size
                    )

                    result = get_position(date, lons[j], lats[j])
                    spatial_position = torch.tensor(
                        [result["azimuth"], result["altitude"], lats[j] / 10.0],
                        device=device,
                    )
                    context_global_next = torch.cat(
                        [
                            spatial_position.unsqueeze(0),
                            logsnr_t_next.unsqueeze(-1) / 10.0,
                        ],
                        dim=1,
                    )

                    patch_x_euler_batch.append(patch_x_euler)
                    patch_context_batch.append(patch_context)
                    context_global_next_batch.append(context_global_next)

                model_input_next = torch.cat(
                    [
                        torch.cat(patch_context_batch, dim=0),
                        torch.cat(patch_x_euler_batch, dim=0),
                    ],
                    dim=2,
                )
                pred_next_batch = model(
                    model_input_next.float(),
                    torch.cat(context_global_next_batch, dim=0).float(),
                )[:, :, 4:, :, :]

                for j, (x_start, y_start) in enumerate(coords_batch):
                    aggregated_pred_next[
                        ...,
                        y_start : y_start + patch_size,
                        x_start : x_start + patch_size,
                    ] += pred_next_batch[j : j + 1]

        averaged_pred_next = aggregated_pred_next / overlap_counts
        v2 = averaged_pred_next
        x_t_full_res = x_t_full_res - (dt / 2) * (averaged_velocity + v2)
        x_t_full_res = x_t_full_res.clamp(-7, 7)
    # Always add back the last context since always forecasting residual
    # Assuming initial_context is already normalized and the model expects residual.
    # The initial_context is (B, C, 4, H_small, W_small).
    # The last frame of context is initial_context[:, :, 3:4].
    # This needs to be upsampled to H_big, W_big if initial_context itself is small.
    # The last frame of context is initial_context[:, :, 3:4], which is (1, C, 1, H_big, W_big).
    last_context_frame = initial_context[:, :, 3:4]

    # Expand it to match the 2-frame generated output
    last_context_frame_expanded = last_context_frame.expand(-1, -1, 2, -1, -1)

    x_t_full_res = x_t_full_res + last_context_frame_expanded

    return x_t_full_res.cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Tiled Diffusion for large image forecasting with Rectified Flow."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/epoch_146_rectified_flow.safetensors",
        help="Path to the pre-trained model .safetensors file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_forecasts",
        help="Directory to save generated forecasts.",
    )
    parser.add_argument(
        "--forecast_steps",
        type=int,
        default=12,
        help="Number of autoregressive forecast steps to generate (total frames).",
    )
    parser.add_argument(
        "--target_H", type=int, default=2540, help="Target height for the large image."
    )
    parser.add_argument(
        "--target_W", type=int, default=4132, help="Target width for the large image."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data_inference/2025081108",
        help="Directory containing the .npy files for initial context.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=128,
        help="Size of patches trained on (e.g., 128 for 128x128).",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        default=100,
        help="Number of denoising steps for each tiled diffusion process.",
    )
    parser.add_argument(
        "--model_channels",
        type=int,
        default=12,
        help="Number of input/output channels for the model.",
    )
    parser.add_argument(
        "--model_features",
        default=[64, 128, 256],  # ,
        help="List of feature counts for the U-Net architecture.",
    )
    parser.add_argument(
        "--num_additional_resnet_blocks",
        type=int,
        default=2,
        help="number of additional resnet block.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for processing patches during inference.",
    )
    parser.add_argument(
        "--context_dim",
        type=int,
        default=4,
        help="Dimension of the context for FiLM conditioning.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Dimension of the embedding for FiLM conditioning.",
    )
    parser.add_argument(
        "--context_frames",
        type=int,
        default=4,
        help="Number of context frames expected by the model.",
    )
    parser.add_argument(
        "--parametrization",
        type=str,
        default="standard",
        choices=["standard", "endpoint"],
        help="Parametrization used in the Rectified Flow model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    model = UNet3D(
        in_channels=args.model_channels,
        out_channels=args.model_channels,
        features=args.model_features,
        context_dim=args.context_dim,
        embedding_dim=args.embedding_dim,
        context_frames=args.context_frames,
        num_additional_resnet_blocks=args.num_additional_resnet_blocks,
    )

    # Load model weights
    if os.path.exists(args.model_path):
        loaded_state_dict = load_file(args.model_path)
        model.load_state_dict(loaded_state_dict)
        print(f"Loaded model weights from {args.model_path}")
    else:
        print(
            f"Warning: Model weights not found at {args.model_path}. Using randomly initialized model."
        )

    model.to(args.device)

    # Set initial date for predictions (pick a summer date for sun position)
    initial_date = datetime(2025, 8, 11, 8, 0)

    # --- Autoregressive Generation Loop ---

    # Load initial context from data
    data_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".npy")])
    if len(data_files) < args.context_frames:
        raise ValueError(
            f"Not enough data files in {args.data_dir}. Need at least {args.context_frames}, found {len(data_files)}"
        )

    initial_frames = []
    for i in range(args.context_frames):
        file_path = os.path.join(args.data_dir, data_files[i])
        frame = np.load(file_path)  # Shape: (1, C, H, W)
        initial_frames.append(frame)

    # Stack along time dimension: (1, C, T, H, W)
    current_high_res_context = np.stack(initial_frames, axis=2)
    current_high_res_context = (
        torch.from_numpy(current_high_res_context).float().to(args.device)
    )
    current_high_res_context = normalize(
        current_high_res_context, args.device
    )  # Normalize initial true data

    # Each step of autoregressive generation predicts 2 frames.
    # We need to run (forecast_steps / 2) iterations.
    for step_idx in range(args.forecast_steps // 2):
        print(f"Autoregressive step {step_idx + 1}/{args.forecast_steps // 2}")

        # Compute the date for the predicted frames
        prediction_date = initial_date + timedelta(minutes=10 * (step_idx * 2 + 1))

        # Perform tiled inference for the next 2 frames
        # The `initial_context` for `tiled_inference` is now `current_high_res_context`
        # which acts as the fixed conditioning input for predicting the *next 2 frames*.
        generated_two_frames = tiled_inference(
            model=model,
            initial_context=current_high_res_context,
            patch_size=args.patch_size,
            steps=args.denoising_steps,
            device=args.device,
            date=prediction_date,
            parametrization=args.parametrization,
            nb_channels=args.model_channels,
            normalize_func=normalize,
            batch_size=args.batch_size,
        )  # Shape: (1, C, 2, H_big, W_big)

        # Save each of the two generated frames independently
        frame1 = generated_two_frames[:, :, 0, :, :]
        frame2 = generated_two_frames[:, :, 1, :, :]

        date1 = prediction_date
        date2 = prediction_date + timedelta(minutes=10)

        filename1 = f"forecast_{date1.strftime('%Y%m%d%H%M')}.npz"
        output_filepath1 = os.path.join(args.output_dir, filename1)
        np.savez_compressed(output_filepath1, forecast=frame1.squeeze(0).numpy())
        print(f"Saved forecast to {output_filepath1}")

        filename2 = f"forecast_{date2.strftime('%Y%m%d%H%M')}.npz"
        output_filepath2 = os.path.join(args.output_dir, filename2)
        np.savez_compressed(output_filepath2, forecast=frame2.squeeze(0).numpy())
        print(f"Saved forecast to {output_filepath2}")

        # Update current_high_res_context for the next autoregressive step
        # Shift the context: remove the oldest 2 frames, add the newly generated 2 frames
        current_high_res_context = torch.cat(
            [
                current_high_res_context[:, :, 2:, :, :],
                generated_two_frames.to(args.device),
            ],
            dim=2,
        )  # (1, C, 4, H_big, W_big)

        # Optional: Save intermediate forecast or visualize (e.g., save every 10 steps)
        # For simplicity, we'll save the final full forecast.

    print("Tiled diffusion and autoregressive forecasting complete.")


if __name__ == "__main__":
    main()
