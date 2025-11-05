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
    full_image_generation  # For potential patch-level eval if needed, but use tiled
)

def compute_reward(model, data_file, initial_date_str, horizons, device, patch_size=128, 
                   denoising_steps=8, batch_size=32, time_step_minutes=10, use_residual=True):
    """
    Compute reward as -sum(MAEs across horizons) using horizon evaluation.
    Assumes single data file for simplicity; average over multiple if val_dir provided.
    """
    results = evaluate_horizons(
        model, data_file, datetime.strptime(initial_date_str, "%Y-%m-%d %H:%M"),
        horizons, device=device, patch_size=patch_size, denoising_steps=denoising_steps,
        batch_size=batch_size, use_residual=use_residual, time_step_minutes=time_step_minutes
    )

    total_mse = sum(metrics['sat_mse'] / key for key, metrics in results.items()) # + metrics['light_mae']

    reward = -total_mse / len(horizons)  # Negative for minimization; normalize by num horizons
    return reward, results

def es_fine_tune(model, val_data_dir, initial_date_str, horizons, T=200, N=30, sigma=0.001, 
                 alpha=0.005, device="cuda", patch_size=128, denoising_steps=16, 
                 batch_size=64, time_step_minutes=10, use_residual=True, save_path=None):
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
        
        for n in range(N):
            torch.manual_seed(seeds[n])
            # Perturb in-place
            for name, param in model.named_parameters():
                noise = torch.randn_like(param, device=device) * sigma
                param.data.add_(noise)
            
            # Evaluate: sample a val file, compute reward
            val_file = random.choice(val_files)
            reward, _ = compute_reward(
                model, str(val_file), initial_date_str, horizons, device, patch_size,
                denoising_steps, batch_size, time_step_minutes, use_residual
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
        
        print(f"Iter {t+1}/{T}: Avg reward {rewards.mean().item():.4f}, Std {rewards.std().item():.4f}")
    
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
    parser.add_argument("--T", type=int, default=200, help="ES iterations.")
    parser.add_argument("--N", type=int, default=10, help="Samples per iteration.")
    parser.add_argument("--sigma", type=float, default=0.001, help="Perturbation std.")
    parser.add_argument("--alpha", type=float, default=0.005, help="Update rate.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--denoising_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--time_step_minutes", type=int, default=10)
    parser.add_argument("--output_model_path", type=str, default="fine_tuned_model.safetensors",
                        help="Path to save fine-tuned model.")
    args = parser.parse_args()
    
    model = load_model(args.base_model_path, args.device)
    es_fine_tune(
        model, args.val_data_dir, args.initial_date_str, args.horizons,
        T=args.T, N=args.N, sigma=args.sigma, alpha=args.alpha,
        device=args.device, patch_size=args.patch_size, denoising_steps=args.denoising_steps,
        batch_size=args.batch_size, time_step_minutes=args.time_step_minutes,
        save_path=args.output_model_path
    )

if __name__ == "__main__":
    main()
