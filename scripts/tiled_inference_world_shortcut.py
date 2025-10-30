import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
import sys
from datetime import datetime, timedelta
from suncalc import get_position
import pyproj
import h5py
import yaml

# Add project root to sys.path
project_root = os.path.abspath("/workspace/meteolibre_model/")
sys.path.insert(0, project_root)
# Load config
config_path = os.path.join(project_root, "meteolibre_model/config/configs.yml")
with open(config_path) as f:
    config = yaml.safe_load(f)
params = config["model_v0_mtg_world_lightning_shortcut"]

# Constants for coordinates (will be set from HDF5 file)
from meteolibre_model.models.unet3d_film_dual import DualUNet3DFiLM
from meteolibre_model.diffusion.rectified_flow_lightning_shortcut import (
    normalize,
    denormalize,
    CLIP_MIN,
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
    c_sat,
    c_lightning,
    transform,
    epsg,
    transformer,
    batch_size=64,
):
    model.eval()
    model.to(device)
    _, C, T_ctx, H_big, W_big = initial_context.shape
    x_t_full_res = torch.randn(1, C, 1, H_big, W_big, device=device)
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
    d_const = 1.0 / steps
    for i in tqdm(range(steps), desc="Tiled Denoising"):
        t_val = 1.0 - i * d_const
        t_batch_val = torch.full((1,), t_val, device=device)
        d_batch_val = torch.full((1,), d_const, device=device)
        aggregated_pred = torch.zeros_like(x_t_full_res, device=device)
        overlap_counts = torch.zeros_like(x_t_full_res, device=device)
        with torch.no_grad():
            for i_batch in range(0, len(patch_coords), batch_size):
                coords_batch = patch_coords[i_batch : i_batch + batch_size]
                patch_x_t_batch, patch_context_batch, context_global_batch = [], [], []
                pixel_xs = [x + patch_size // 2 for x, y in coords_batch]
                pixel_ys = [y + patch_size // 2 for x, y in coords_batch]
                lons, lats = [], []
                for j in range(len(coords_batch)):
                    px = pixel_xs[j]
                    py = pixel_ys[j]
                    x_crs = transform[0] * px + transform[1] * py + transform[2]
                    y_crs = transform[3] * px + transform[4] * py + transform[5]
                    if epsg != 4326:
                        lon, lat = transformer.transform(x_crs, y_crs)
                    else:
                        lon, lat = x_crs, y_crs
                    lons.append(lon)
                    lats.append(lat)
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
                        [
                            spatial_position.unsqueeze(0),
                            t_batch_val.unsqueeze(-1),
                            d_batch_val.unsqueeze(-1),
                        ],
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
                model_input_sat = model_input[:, :c_sat]
                model_input_lightning = model_input[:, c_sat : (c_sat + c_lightning)]
                sat_pred_batch, lightning_pred_batch = model(
                    model_input_sat.float(),
                    model_input_lightning.float(),
                    torch.cat(context_global_batch, dim=0).float(),
                )
                pred_batch = torch.cat([sat_pred_batch, lightning_pred_batch], dim=1)[
                    :, :, 4:, :, :
                ]
                for j, (x_start, y_start) in enumerate(coords_batch):
                    aggregated_pred[
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
        averaged_pred = aggregated_pred / overlap_counts
        x_t_full_res = x_t_full_res - averaged_pred * d_const
        x_t_full_res = x_t_full_res.clamp(-7, 7)
    # Always add back the last context since always forecasting residual
    last_context_frame = initial_context[:, :, 3:4]
    x_t_full_res = x_t_full_res + last_context_frame
    # Clamp results to be within the expected normalized range
    x_t_full_res = torch.where(
        last_context_frame == CLIP_MIN, last_context_frame, x_t_full_res
    )
    return x_t_full_res.cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Tiled Diffusion for large image forecasting with Shortcut Rectified Flow."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/20251019_epoch61_mtg_lightning_soap.safetensors",
        help="Path to the pre-trained model .safetensors file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_forecasts_shortcut",
        help="Directory to save generated forecasts.",
    )
    parser.add_argument(
        "--forecast_steps",
        type=int,
        default=18,
        help="Number of autoregressive forecast steps to generate.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="../dataset/data_inference_full/2025-10-14_04-00_full.h5",
        help="Path to the HDF5 file containing the initial context.",
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
        default=128,
        help="Number of denoising steps for each tiled diffusion process.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=params["batch_size"],
        help="Batch size for processing patches during inference.",
    )
    parser.add_argument(
        "--context_frames",
        type=int,
        default=params["model"]["context_frames"],
        help="Number of context frames expected by the model.",
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
    model_params = params["model"]
    model = DualUNet3DFiLM(**model_params)
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
    # Load initial context from HDF5 file
    if not os.path.exists(args.data_file):
        raise ValueError(f"Data file {args.data_file} not found.")
    with h5py.File(args.data_file, "r") as hf:
        sat_data = hf["sat_data"][:]
        lightning_data = hf["lightning_data"][:]
        num_frames = hf.attrs["num_frames"]
        target_H = hf.attrs["target_height"]
        target_W = hf.attrs["target_width"]
        transform = hf.attrs["transform"]
        epsg = hf.attrs["epsg"]
        c_sat = hf.attrs["num_sat_channels"]
        c_lightning = hf.attrs["num_lightning_channels"]
    if epsg != 4326:
        transformer = pyproj.Transformer.from_crs(
            f"EPSG:{epsg}", "EPSG:4326", always_xy=True
        )
    else:
        transformer = None
    nb_channels = c_sat + c_lightning
    # Parse initial_date from filename
    filename = os.path.basename(args.data_file)
    # Assuming filename like "2025-10-14_04-00_full.h5"
    date_str = filename.split("_full.h5")[0]  # "2025-10-14_04-00"
    date_part, time_part = date_str.split("_")[0], date_str.split("_")[1]
    year, month, day = map(int, date_part.split("-"))
    hour, minute = map(int, time_part.split("-"))
    initial_date = datetime(year, month, day, hour, minute) - timedelta(minutes=18 * 10)

    if num_frames < args.context_frames:
        raise ValueError(
            f"Not enough frames in {args.data_file}. Need at least {args.context_frames}, found {num_frames}"
        )
    initial_frames = []
    for i in range(args.context_frames):
        sat_frame = sat_data[i]  # (C, H, W)
        lightning_frame = lightning_data[i]  # (1, H, W)
        frame = np.concatenate([sat_frame, lightning_frame], axis=0)[
            None, ...
        ]  # (1, nb_channels, H, W)
        initial_frames.append(frame)
    # Stack along time dimension: (1, nb_channels, T, H, W)
    current_high_res_context = np.stack(initial_frames, axis=2)
    current_high_res_context = (
        torch.from_numpy(current_high_res_context).float().to(args.device)
    )
    # Split into sat and lightning
    sat_data = current_high_res_context[:, :c_sat]
    lightning_data = current_high_res_context[:, c_sat:]
    # Normalize
    sat_data, lightning_data = normalize(sat_data, lightning_data, args.device)
    # Concat back
    current_high_res_context = torch.cat([sat_data, lightning_data], dim=1)
    # Autoregressive Generation Loop
    for step_idx in range(args.forecast_steps):
        print(f"Autoregressive step {step_idx + 1}/{args.forecast_steps}")
        # Compute the date for the predicted frame
        prediction_date = initial_date + timedelta(minutes=10 * (step_idx + 1))
        # Perform tiled inference for the next frame
        generated_frame = tiled_inference(
            model=model,
            initial_context=current_high_res_context,
            patch_size=args.patch_size,
            steps=args.denoising_steps,
            device=args.device,
            date=prediction_date,
            c_sat=c_sat,
            c_lightning=c_lightning,
            batch_size=args.batch_size,
            transform=transform,
            epsg=epsg,
            transformer=transformer,
        )  # Shape: (1, nb_channels, 1, H_big, W_big)
        # Split and denormalize
        sat_generated = generated_frame[:, :c_sat]
        lightning_generated = generated_frame[:, c_sat:]
        sat_generated, lightning_generated = denormalize(
            sat_generated, lightning_generated, args.device
        )
        # Save the generated frame (sat and lightning channels)
        sat_frame = sat_generated[:, :, 0, :, :]
        lightning_frame = lightning_generated[:, :, 0, :, :]
        date = prediction_date
        filename = f"forecast_{date.strftime('%Y%m%d%H%M')}.npz"
        output_filepath = os.path.join(args.output_dir, filename)
        np.savez_compressed(
            output_filepath,
            sat_forecast=sat_frame.squeeze(0).cpu().numpy(),
            lightning_forecast=lightning_frame.squeeze(0).cpu().numpy(),
        )
        print(f"Saved forecast to {output_filepath}")
        # Update current_high_res_context for the next autoregressive step
        # Shift the context: remove the oldest frame, add the newly generated frame
        current_high_res_context = torch.cat(
            [
                current_high_res_context[:, :, 1:, :, :],
                generated_frame.to(args.device),
            ],
            dim=2,
        )  # (1, nb_channels, 4, H_big, W_big)
    print("Tiled diffusion and autoregressive forecasting complete.")


if __name__ == "__main__":
    main()
