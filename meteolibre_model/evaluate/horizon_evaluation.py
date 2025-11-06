import torch
import torch.nn.functional as F
import numpy as np
import h5py
import yaml
import os
import sys
from datetime import datetime, timedelta
from tqdm import tqdm
import pyproj
from suncalc import get_position
from safetensors.torch import load_file

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Load config
config_path = os.path.join(project_root, "meteolibre_model/config/configs.yml")
with open(config_path) as f:
    config = yaml.safe_load(f)
params = config["model_v1_mtg_world_lightning_shortcut"]

from meteolibre_model.models.unet3d_film_dual import DualUNet3DFiLM
from meteolibre_model.diffusion.rectified_flow_lightning_shortcut import (
    normalize, CLIP_MIN
)

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
    use_residual=True,
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
        last_context_frame = initial_context[:, :, 3:4]
    if use_residual:
        x_t_full_res = x_t_full_res + last_context_frame
    # Clamp results to be within the expected normalized range
    x_t_full_res = torch.where(
        last_context_frame == CLIP_MIN, last_context_frame, x_t_full_res
    )
    return x_t_full_res.cpu()

def evaluate_horizons(
    model,
    data_file=None,
    initial_date=None,  # datetime object for the first forecast horizon (h=1)
    horizons=None,  # list of integers [1,2,3,...]
    device="cuda",
    patch_size=128,
    denoising_steps=128,
    batch_size=64,
    context_frames=4,
    use_residual=True,
    time_step_minutes=10,
    subgrid_size=None,
    seed=None,
    baseline=False,
    initial_context=None,
    gts=None,
    fixed_params=None,  # dict with 'c_sat', 'c_lightning', 'transform', 'epsg', 'transformer'
):
    """
    Evaluate model performance at specified horizons using tiled inference.
    
    Args:
        model: Loaded model in eval mode.
        data_file: Path to HDF5 file with sufficient frames.
        initial_date: datetime for the first forecast (horizon 1).
        horizons: List of forecast horizons to evaluate.
        device: Device to use.
        patch_size: Patch size for tiled inference.
        denoising_steps: Number of denoising steps.
        batch_size: Batch size for patches.
        context_frames: Number of context frames.
        use_residual: Whether to use residual forecasting.
        time_step_minutes: Time step between frames in minutes.
        subgrid_size: Size for random subgrid cropping if not None.
        seed: Random seed for reproducible subgrid cropping.
        baseline: If True, compute persistence baseline metrics.
    
    Returns:
        Dict with horizons as keys and metrics as values.
        Metrics: {'sat_mse': float, 'sat_mae': float, 'light_mse': float, 'light_mae': float}
    """
    if baseline:
        model = None  # Not used for baseline
    
    if not baseline:
        model.to(device)
        model.eval()
    
    with torch.no_grad():
        max_h = max(horizons)
        
        if data_file is not None:
            # Load HDF5 data
            with h5py.File(data_file, "r") as hf:
                sat_data_full = np.array(hf["sat_data"])  # (F, C_sat, H_full, W_full)
                lightning_data_full = np.array(hf["lightning_data"])  # (F, 1, H_full, W_full)
                num_frames = hf.attrs["num_frames"]
                H_full = hf.attrs["target_height"]
                W_full = hf.attrs["target_width"]
                transform_full = hf.attrs["transform"]
                epsg = hf.attrs["epsg"]
                c_sat = hf.attrs["num_sat_channels"]
                c_lightning = hf.attrs["num_lightning_channels"]
            
            if subgrid_size is not None:
                if seed is not None:
                    np.random.seed(seed)
                    crop_y = np.random.randint(0, H_full - subgrid_size + 1)
                    crop_x = np.random.randint(0, W_full - subgrid_size + 1)
                else:
                    crop_y = (H_full - subgrid_size) // 2
                    crop_x = (W_full - subgrid_size) // 2
                sat_data = sat_data_full[:, :, crop_y:crop_y+subgrid_size, crop_x:crop_x+subgrid_size]
                lightning_data = lightning_data_full[:, :, crop_y:crop_y+subgrid_size, crop_x:crop_x+subgrid_size]
                H = subgrid_size
                W = subgrid_size
                # Adjust transform for crop
                a, b, c, d, e, f = transform_full
                new_c = a * crop_x + b * crop_y + c
                new_f = d * crop_x + e * crop_y + f
                transform = [a, b, new_c, d, e, new_f]
            else:
                sat_data = sat_data_full
                lightning_data = lightning_data_full
                H = H_full
                W = W_full
                transform = transform_full
            
            if num_frames < context_frames + max_h:
                raise ValueError(
                    f"Insufficient frames in {data_file}: need at least {context_frames + max_h}, "
                    f"got {num_frames}. Ensure HDF5 has context + max horizon frames."
                )
            
            if epsg != 4326:
                transformer = pyproj.Transformer.from_crs(
                    f"EPSG:{epsg}", "EPSG:4326", always_xy=True
                )
            else:
                transformer = None
            
            # Build initial context from frames 0 to context_frames-1
            initial_frames = []
            for i in range(context_frames):
                sat_frame = sat_data[i]  # (C_sat, H, W)
                lightning_frame = lightning_data[i]  # (1, H, W)
                frame = np.concatenate([sat_frame, lightning_frame], axis=0)  # (C_total, H, W)
                initial_frames.append(frame[None, ...])  # (1, C, H, W)
            
            context_np = np.stack(initial_frames, axis=2)  # (1, C, T_ctx, H, W)
            current_context = torch.from_numpy(context_np).float().to(device)
            
            # Normalize context
            sat_ctx = current_context[:, :c_sat]
            light_ctx = current_context[:, c_sat:]
            sat_ctx, light_ctx = normalize(sat_ctx, light_ctx, device)
            current_context = torch.cat([sat_ctx, light_ctx], dim=1)
            
            # Last context frame for persistence
            last_context_frame = current_context[:, :, -1:, :, :]  # (1, C, 1, H, W)
            sat_last = last_context_frame[:, :c_sat, 0]  # (1, C_sat, H, W)
            light_last = last_context_frame[:, c_sat:, 0]  # (1, C_lightning, H, W)
            
            # Prepare ground truths (normalized)
            gts_dict = {}
            for h in horizons:
                frame_idx = context_frames + h - 1
                sat_gt = sat_data[frame_idx]  # (C_sat, H, W)
                light_gt = lightning_data[frame_idx]  # (1, H, W)
                
                # Normalize (add batch and time dims)
                sat_gt_t = torch.from_numpy(sat_gt).float().to(device).unsqueeze(0).unsqueeze(2)  # (1, C_sat, 1, H, W)
                light_gt_t = torch.from_numpy(light_gt).float().to(device).unsqueeze(0).unsqueeze(2)  # (1, 1, 1, H, W)
                sat_gt_norm, light_gt_norm = normalize(sat_gt_t, light_gt_t, device)
                
                gts_dict[h] = {
                    'sat': sat_gt_norm.squeeze(0).squeeze(1),  # (C_sat, H, W)
                    'lightning': light_gt_norm.squeeze(0).squeeze(1)  # (1, H, W)
                }
            
            fixed_params = {
                'c_sat': c_sat,
                'c_lightning': c_lightning,
                'transform': transform,
                'epsg': epsg,
                'transformer': transformer
            }
        else:
            if initial_context is None or gts is None or fixed_params is None:
                raise ValueError("When data_file is None, initial_context, gts, and fixed_params must be provided.")
            current_context = initial_context.to(device)
            c_sat = fixed_params['c_sat']
            c_lightning = fixed_params['c_lightning']
            transform = fixed_params['transform']
            epsg = fixed_params['epsg']
            transformer = fixed_params['transformer']
            gts_dict = gts
            last_context_frame = current_context[:, :, -1:, :, :]  # (1, C, 1, H, W)
            sat_last = last_context_frame[:, :c_sat, 0]  # (1, C_sat, H, W)
            light_last = last_context_frame[:, c_sat:, 0]  # (1, C_lightning, H, W)
        
        # Compute metrics
        results = {}
        if baseline:
            for h in horizons:
                gt_sat = gts[h]['sat'].unsqueeze(0)  # (1, C_sat, H, W)
                gt_light = gts[h]['lightning'].unsqueeze(0)  # (1, 1, H, W)
                
                sat_pred = sat_last
                light_pred = light_last

                sat_mse = F.mse_loss(sat_pred, gt_sat)
                sat_mae = F.l1_loss(sat_pred, gt_sat)
                light_mse = F.mse_loss(light_pred, gt_light)
                light_mae = F.l1_loss(light_pred, gt_light)
                
                results[h] = {
                    'sat_mse': sat_mse.item(),
                    'sat_mae': sat_mae.item(),
                    'light_mse': light_mse.item(),
                    'light_mae': light_mae.item(),
                }
                print(f"Baseline Horizon {h}: sat_mse={sat_mse.item():.4f}, light_mse={light_mse.item():.4f}")
        else:
            # Autoregressive generation
            current_date = initial_date
            for step in range(1, max_h + 1):
                print(f"Generating step {step}/{max_h}")
                
                generated_frame = tiled_inference(
                    model=model,
                    initial_context=current_context,
                    patch_size=patch_size,
                    steps=denoising_steps,
                    device=device,
                    date=current_date,
                    c_sat=c_sat,
                    c_lightning=c_lightning,
                    transform=transform,
                    epsg=epsg,
                    transformer=transformer,
                    batch_size=batch_size,
                    use_residual=use_residual,
                )  # (1, C, 1, H, W) normalized
                
                # Split generated frame
                sat_gen = generated_frame[:, :c_sat, 0]  # (1, C_sat, H, W)
                light_gen = generated_frame[:, c_sat:, 0]  # (1, C_lightning, H, W)
                
                if step in horizons:
                    gt_sat = gts[step]['sat'].unsqueeze(0)  # (1, C_sat, H, W)
                    gt_light = gts[step]['lightning'].unsqueeze(0)  # (1, 1, H, W)
                    
                    sat_gen = sat_gen.to(device)
                    gt_sat = gt_sat.to(device)
                    light_gen = light_gen.to(device)
                    gt_light = gt_light.to(device)

                    sat_mse = F.mse_loss(sat_gen, gt_sat)
                    sat_mae = F.l1_loss(sat_gen, gt_sat)
                    light_mse = F.mse_loss(light_gen, gt_light)
                    light_mae = F.l1_loss(light_gen, gt_light)
                    
                    results[step] = {
                        'sat_mse': sat_mse.item(),
                        'sat_mae': sat_mae.item(),
                        'light_mse': light_mse.item(),
                        'light_mae': light_mae.item(),
                    }
                    print(f"Horizon {step}: sat_mse={sat_mse.item():.4f}, light_mse={light_mse.item():.4f}")
                
                # Update context for next step: shift and append generated frame
                generated_frame = generated_frame.unsqueeze(2) if generated_frame.dim() == 4 else generated_frame  # Ensure (1, C, 1, H, W)
                current_context = torch.cat([
                    current_context[:, :, 1:, :, :].to(device),  # Remove first time step
                    generated_frame.to(device)  # Add new frame
                ], dim=2)
                
                # Update date
                current_date += timedelta(minutes=time_step_minutes)
        
        return results

def load_model(model_path, device="cuda"):
    """
    Load the model from safetensors file.
    
    Args:
        model_path: Path to .safetensors file.
        device: Device to load on.
    
    Returns:
        Loaded model.
    """
    model_params = params["model"]
    model = DualUNet3DFiLM(**model_params)
    if os.path.exists(model_path):
        state_dict = load_file(model_path)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}, using random initialization.")
    model.to(device)
    return model

def quick_evaluate(
    model_path,
    data_file,
    initial_date_str,  # e.g., "2025-10-14 04:00"
    horizons=[1, 2, 3, 6, 12, 18],
    device="cuda",
    patch_size=128,
    denoising_steps=64,
    batch_size=32,
    output_dir=None,
    baseline=False,
):
    """
    Quick evaluation wrapper.
    
    Args:
        model_path: Path to model.
        data_file: HDF5 data file.
        initial_date_str: Initial date string for first forecast.
        horizons: List of horizons.
        ... other params.
        baseline: If True, compute persistence baseline.
    
    Returns:
        Evaluation results dict.
    """
    if baseline:
        if model_path is not None:
            print("Warning: model_path provided but ignored for baseline evaluation.")
        model = None
    else:
        if model_path is None:
            raise ValueError("model_path is required for non-baseline evaluation.")
        model = load_model(model_path, device)
    
    initial_date = datetime.strptime(initial_date_str, "%Y-%m-%d %H:%M")
    
    results = evaluate_horizons(
        model,
        data_file,
        initial_date,
        horizons,
        device=device,
        patch_size=patch_size,
        denoising_steps=denoising_steps,
        batch_size=batch_size,
        baseline=baseline,
    )
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.savez(os.path.join(output_dir, "evaluation_results.npz"), **{str(k): v for k, v in results.items()})
        print(f"Results saved to {output_dir}/evaluation_results.npz")
    
    return results

# Example usage:
# results = quick_evaluate(
#     model_path="models/your_model.safetensors",
#     data_file="path/to/data.h5",
#     initial_date_str="2025-10-14 04:00",
#     horizons=[1, 3, 6],
#     denoising_steps=32
# )
# print(results)
