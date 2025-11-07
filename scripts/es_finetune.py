import argparse
import os
import sys
import torch
import torch.nn.functional as F
import random
import yaml
from datetime import datetime, timedelta
from tqdm import tqdm
import safetensors.torch
from pathlib import Path
import math

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Load config
config_path = os.path.join(project_root, "meteolibre_model/config/configs.yml")
with open(config_path) as f:
    config = yaml.safe_load(f)
params = config["model_v1_mtg_world_lightning_shortcut"]  # Typo fix if needed: model_v1_mtg_world_lightning_shortcut

from meteolibre_model.models.unet3d_film_dual import DualUNet3DFiLM
from meteolibre_model.evaluate.horizon_evaluation import (
    load_model, evaluate_horizons, normalize, CLIP_MIN, tiled_inference
)
from meteolibre_model.diffusion.rectified_flow_lightning_shortcut import (
    full_image_generation, normalize, CLIP_MIN  # For potential patch-level eval if needed, but use tiled
)

import h5py
import numpy as np
import pyproj

from torch.utils.tensorboard import SummaryWriter

def compute_reward(model, data_file, initial_date_str, horizons, device, patch_size=128, 
                   denoising_steps=8, batch_size=32, time_step_minutes=10, use_residual=True, subgrid_size=None, seed=None,
                   initial_context=None, gts=None, fixed_params=None):
    """
    Compute reward as -sum(MAEs across horizons) using horizon evaluation.
    Assumes single data file for simplicity; average over multiple if val_dir provided.
    """
    initial_date = datetime.strptime(initial_date_str, "%Y-%m-%d %H:%M")
    if initial_context is not None:
        results = evaluate_horizons(
            model, data_file=None, initial_date=initial_date,
            horizons=horizons, device=device, patch_size=patch_size, denoising_steps=denoising_steps,
            batch_size=batch_size, use_residual=use_residual, time_step_minutes=time_step_minutes,
            subgrid_size=subgrid_size, seed=seed,
            initial_context=initial_context, gts=gts, fixed_params=fixed_params
        )
    else:
        results = evaluate_horizons(
            model, data_file, initial_date,
            horizons, device=device, patch_size=patch_size, denoising_steps=denoising_steps,
            batch_size=batch_size, use_residual=use_residual, time_step_minutes=time_step_minutes,
            subgrid_size=subgrid_size, seed=seed
        )

    total_mse = sum(metrics['sat_mse'] / math.sqrt(key) for key, metrics in results.items()) # + metrics['light_mae']

    reward = -total_mse / len(horizons)  # Negative for minimization; normalize by num horizons
    return reward, results

def load_subgrid_data(val_file, seed_data, horizons, device, context_frames, subgrid_size=500):
    """
    Load and preprocess HDF5 data for subgrid evaluation in ES.
    Returns initial_context (normalized tensor), gts (dict), fixed_params (dict).
    """
    import h5py
    import numpy as np
    import pyproj
    
    with h5py.File(str(val_file), "r") as hf:
        sat_data_full = np.array(hf["sat_data"])
        lightning_data_full = np.array(hf["lightning_data"])
        num_frames = hf.attrs["num_frames"]
        H_full = hf.attrs["target_height"]
        W_full = hf.attrs["target_width"]
        transform_full = hf.attrs["transform"]
        epsg = hf.attrs["epsg"]
        c_sat = hf.attrs["num_sat_channels"]
        c_lightning = hf.attrs["num_lightning_channels"]
    
    np.random.seed(seed_data)
    crop_y = np.random.randint(0, H_full - subgrid_size + 1)
    crop_x = np.random.randint(0, W_full - subgrid_size + 1)
    sat_data = sat_data_full[:, :, crop_y:crop_y+subgrid_size, crop_x:crop_x+subgrid_size]
    lightning_data = lightning_data_full[:, :, crop_y:crop_y+subgrid_size, crop_x:crop_x+subgrid_size]
    H = subgrid_size
    W = subgrid_size
    # Adjust transform for crop
    a, b, c, d, e, f = transform_full
    new_c = a * crop_x + b * crop_y + c
    new_f = d * crop_x + e * crop_y + f
    transform = [a, b, new_c, d, e, new_f]
    
    max_h = max(horizons)
    if num_frames < context_frames + max_h:
        raise ValueError(f"Insufficient frames in {val_file}: need at least {context_frames + max_h}, got {num_frames}.")
    
    if epsg != 4326:
        transformer = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    else:
        transformer = None
    
    # Build initial context
    initial_frames = []
    for i in range(context_frames):
        sat_frame = sat_data[i]
        lightning_frame = lightning_data[i]
        frame = np.concatenate([sat_frame, lightning_frame], axis=0)
        initial_frames.append(frame[None, ...])
    
    context_np = np.stack(initial_frames, axis=2)
    current_context = torch.from_numpy(context_np).float().to(device)
    
    # Normalize context
    sat_ctx = current_context[:, :c_sat]
    light_ctx = current_context[:, c_sat:]
    sat_ctx, light_ctx = normalize(sat_ctx, light_ctx, device)
    initial_context = torch.cat([sat_ctx, light_ctx], dim=1)
    
    # Prepare ground truths
    gts = {}
    for h in horizons:
        frame_idx = context_frames + h - 1
        sat_gt = sat_data[frame_idx]
        light_gt = lightning_data[frame_idx]
        
        sat_gt_t = torch.from_numpy(sat_gt).float().to(device).unsqueeze(0).unsqueeze(2)
        light_gt_t = torch.from_numpy(light_gt).float().to(device).unsqueeze(0).unsqueeze(2)
        sat_gt_norm, light_gt_norm = normalize(sat_gt_t, light_gt_t, device)
        
        gts[h] = {
            'sat': sat_gt_norm.squeeze(0).squeeze(1),
            'lightning': light_gt_norm.squeeze(0).squeeze(1)
        }
    
    fixed_params = {
        'c_sat': c_sat,
        'c_lightning': c_lightning,
        'transform': transform,
        'epsg': epsg,
        'transformer': transformer
    }
    
    return initial_context, gts, fixed_params


def es_fine_tune(model, val_data_dir, initial_date_str, horizons, T=200, N=30, sigma=0.001, 
                 alpha=0.005, device="cuda", patch_size=128, denoising_steps=16, 
                 batch_size=64, time_step_minutes=10, use_residual=True, save_path=None, writer=None, log_interval=10):
    """
    ES fine-tuning using parameter perturbations and horizon MAE rewards.
    Assumes val_data_dir contains HDF5 files; samples one per evaluation for efficiency.
    Parallelization via torch.distributed can be added later.
    """
    model.to(device)
    model.train()  # For param updates, but eval during perturbations
    
    val_files = list(Path(val_data_dir).glob("*.h5"))
    if not val_files:
        raise ValueError(f"No HDF5 files found in {val_data_dir}")
    
    theta0 = {k: v.clone() for k, v in model.state_dict().items()}  # Backup original
    
    for t in tqdm(range(T), desc="ES Iterations"):
        seeds = [random.randint(0, 2**32 - 1) for _ in range(N)]
        rewards = []
        
        val_file = random.choice(val_files)
        seed_data = random.randint(0, 2**32 - 1)
        
        context_frames = params["model"]["context_frames"]
        initial_context, gts, fixed_params = load_subgrid_data(
            val_file, seed_data, horizons, device, context_frames
        )
        
        for n in range(N):
            torch.manual_seed(seeds[n])
            # Perturb in-place
            for name, param in model.named_parameters():
                noise = torch.randn_like(param, device=device) * sigma
                param.data.add_(noise)
            
            # Evaluate with preloaded data
            reward, _ = compute_reward(
                model, None, initial_date_str, horizons, device, patch_size,
                denoising_steps, batch_size, time_step_minutes, use_residual,
                subgrid_size=500, seed=seed_data,
                initial_context=initial_context, gts=gts, fixed_params=fixed_params
            )
            rewards.append(reward)
            
            # Restore
            torch.manual_seed(seeds[n])
            for name, param in model.named_parameters():
                noise = torch.randn_like(param, device=device) * sigma
                param.data.sub_(noise)
        
        # Normalize rewards to z-scores
        rewards = torch.tensor(rewards, device=device)
        if rewards.std() == 0:
            continue  # Skip if no variance
        z_scores = (rewards - rewards.mean()) / rewards.std()
        
        # Update
        for n in range(N):
            torch.manual_seed(seeds[n])
            weight = alpha * z_scores[n] / N
            for name, param in model.named_parameters():
                noise = torch.randn_like(param, device=device)
                param.data.add_(weight * noise * sigma)  # Scale by sigma as in algo
        
        # Compute and log current model reward after update
        if writer is not None and (t + 1) % log_interval == 0:
            model.eval()
            val_file = random.choice(val_files)
            current_reward, results = compute_reward(
                model, str(val_file), initial_date_str, horizons, device, patch_size,
                denoising_steps, batch_size, time_step_minutes, use_residual,
                subgrid_size=None
            )
            model.train()
            writer.add_scalar('Reward/model_reward', current_reward, t+1)
            print(f"Iter {t+1}/{T}: Model reward {current_reward:.4f}")
            for h in horizons:
                if h in results:
                    writer.add_scalar(f'Metrics/sat_mse_h{h}', results[h]['sat_mse'], t+1)
                    print(f"  Horizon {h}: sat_mse {results[h]['sat_mse']:.4f}")
        
        # Log min/max of perturbation rewards
        if writer is not None:
            writer.add_scalar('Reward/min_perturbation', rewards.min().item(), t+1)
            writer.add_scalar('Reward/max_perturbation', rewards.max().item(), t+1)
        
        print(f"Iter {t+1}/{T}: Avg reward {rewards.mean().item():.4f}, Std {rewards.std().item():.4f}, Min {rewards.min().item():.4f}, Max {rewards.max().item():.4f}")
    
    # Restore if needed or save fine-tuned
    if save_path:
        safetensors.torch.save_file(model.state_dict(), save_path)
        print(f"Fine-tuned model saved to {save_path}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="ES fine-tuning for diffusion model using horizon MAEs.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base .safetensors model.")
    parser.add_argument("--val_data_dir", type=str, required=True, help="Dir with HDF5 val files.")
    parser.add_argument("--initial_date_str", type=str, required=True, 
                        help="Date for eval (e.g., '2025-10-14 04:00').")
    parser.add_argument("--horizons", type=int, nargs='+', default=[1, 2, 6, 12, 18], 
                        help="Horizons for MAE reward.")
    parser.add_argument("--T", type=int, default=400, help="ES iterations.")
    parser.add_argument("--N", type=int, default=15, help="Samples per iteration.")
    parser.add_argument("--sigma", type=float, default=0.001, help="Perturbation std.")
    parser.add_argument("--alpha", type=float, default=0.005, help="Update rate.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--denoising_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--time_step_minutes", type=int, default=10)
    parser.add_argument("--output_model_path", type=str, default="fine_tuned_model.safetensors",
                        help="Path to save fine-tuned model.")
    parser.add_argument("--log_dir", type=str, default=None, help="Path to TensorBoard log directory. If None, creates one with timestamp.")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval for logging full model reward and metrics.")
    args = parser.parse_args()
    
    if args.log_dir is None:
        log_dir = f"runs/es_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    model = load_model(args.base_model_path, args.device)
    
    # Compute initial reward
    val_files = list(Path(args.val_data_dir).glob("*.h5"))
    if not val_files:
        raise ValueError(f"No HDF5 files found in {args.val_data_dir}")
    val_file = random.choice(val_files)
    model.eval()  # Set to eval for initial reward
    initial_reward, _ = compute_reward(
        model, str(val_file), args.initial_date_str, args.horizons, args.device,
        args.patch_size, args.denoising_steps, args.batch_size, args.time_step_minutes, use_residual=True
    )
    model.train()  # Back to train
    writer.add_scalar('Reward/model_reward', initial_reward, 0)
    print(f"Initial reward: {initial_reward:.4f}")
    
    es_fine_tune(
        model, args.val_data_dir, args.initial_date_str, args.horizons,
        T=args.T, N=args.N, sigma=args.sigma, alpha=args.alpha,
        device=args.device, patch_size=args.patch_size, denoising_steps=args.denoising_steps,
        batch_size=args.batch_size, time_step_minutes=args.time_step_minutes,
        save_path=args.output_model_path,
        writer=writer,
        log_interval=args.log_interval
    )
    
    writer.close()

if __name__ == "__main__":
    main()
