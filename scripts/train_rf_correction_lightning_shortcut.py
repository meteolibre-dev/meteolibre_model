"""
Training script for Correction Model using Hugging Face Accelerate with Rectified Flow (shortcut version)
This script trains a correction model using the MeteoFranceDataset and Unet, following the structure of the main training script.
"""

import sys
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from datetime import datetime
import argparse
import yaml
from safetensors.torch import save_file
from safetensors.torch import load_file

from heavyball import ForeachSOAP

# Add project root to sys.path
project_root = os.path.abspath("/workspace/meteolibre_model/")
sys.path.insert(0, project_root)

from meteolibre_model.dataset.dataset_mtg_lightning_7frames import MeteoLibreMapDataset7Frames
from meteolibre_model.diffusion.rectified_flow_lightning_shortcut import (
    full_image_generation,
    normalize,
    denormalize,
)
from meteolibre_model.diffusion.rectified_flow_correction_training import regression_trainer_step, full_correction_generation
from meteolibre_model.models.unet3d_film_dual import DualUNet3DFiLM

# Load config
config_path = os.path.join(project_root, "meteolibre_model/config/configs.yml")
with open(config_path) as f:
    config = yaml.safe_load(f)
params = config['model_v3_mtg_world_lightning_shortcut']

def main(args):
    # Initialize Accelerator with bfloat16 precision and logging
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=".",
    )
    device = accelerator.device

    # Load hyperparameters from config
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    num_epochs = params['num_epochs']
    gen_steps = params.get('gen_steps', 4)
    log_interval = params['log_every_n_steps']
    save_interval = params['save_every_n_epochs']
    gradient_clip_value = params['gradient_clip_value']
    seed = params['seed']
    sigma_noise_input = params.get('sigma_noise_input', 0.0)
    id_run = str(datetime.utcnow())[:19]
    hps = {
        "batch_size": batch_size,
        "learning_rate": learning_rate / 10.,
        "gen_steps": gen_steps,
    }

    # Set seed for reproducibility
    set_seed(seed)

    accelerator.init_trackers(
        "correction_model_training_rectified_flow_" + id_run, config=hps
    )

    # Initialize dataset
    dataset_6_frames = MeteoLibreMapDataset7Frames(
        localrepo=params['dataset_path'], cache_size=8, seed=seed + 1, nb_temporal=6
    )

    # Initialize DataLoader
    dataloader = DataLoader(
        dataset_6_frames,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Load pre-trained forecast model
    model_params = params["model"]
    forecast_model = DualUNet3DFiLM(**model_params)

    print("loading forecasting model")
    state_dict = load_file(args.forecast_model_path)

    forecast_model.load_state_dict(state_dict)

    # Initialize correction model
    model_params_corr = params["model_correction"]
    correction_model = DualUNet3DFiLM(**model_params_corr)

    print("loading forecasting model")
    state_dict = load_file(args.forecast_model_path)

    correction_model.load_state_dict(state_dict)

    # Initialize optimizer
    #optimizer = ForeachSOAP(correction_model.parameters(), lr=learning_rate, foreach=False)
    optimizer = torch.optim.AdamW(correction_model.parameters(), lr=learning_rate)

    # Prepare for distributed training (only correction_model and optimizer)
    forecast_model, correction_model, optimizer, dataloader = accelerator.prepare(forecast_model, correction_model, optimizer, dataloader)

    global_step = 0

    # Training loop
    for epoch in range(num_epochs):
        correction_model.train()
        total_loss = 0.0

        print("start epoch")

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_main_process,
        )
        for batch in progress_bar:
            with accelerator.accumulate(correction_model):
                # Perform training step using the correction logic
                print("start loss compute")
                loss, loss_sat, loss_kpi = regression_trainer_step(
                    forecast_model,
                    correction_model,
                    batch,
                    device,
                    generation_steps=gen_steps
                )
                accelerator.backward(loss)

                # Gradient clipping
                accelerator.clip_grad_norm_(correction_model.parameters(), gradient_clip_value)

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % log_interval == 0:
                    if accelerator.is_main_process:
                        accelerator.log(
                            {"Loss/train": loss.item()},
                            step=global_step,
                        )

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)

        # Log to Accelerate
        if accelerator.is_main_process:
            accelerator.log({"Loss/train_epoch": avg_loss}, step=epoch)
            print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        # Optional: Visualization (adapt if needed)
        if accelerator.is_main_process and epoch % 5 == 0:  # Every 5 epochs
            with torch.no_grad():
                # Use a sample batch for generation visualization
                sample_batch = next(iter(dataloader))

                # Prepare for autoregressive predictions
                b, c_sat, t_dim, h, w = sample_batch["sat_patch_data"].permute(0, 2, 1, 3, 4).shape
                _, c_lightning, _, _, _ = sample_batch["lightning_patch_data"].permute(0, 2, 1, 3, 4).shape

                # Generate pred_4
                batch_gen1 = {
                    "sat_patch_data": sample_batch["sat_patch_data"][:, :4],
                    "lightning_patch_data": sample_batch["lightning_patch_data"][:, :4],
                    "spatial_position": sample_batch["spatial_position"][:, 0, :],
                }
                pred_4_norm, _ = full_image_generation(
                    forecast_model, batch_gen1, steps=gen_steps, device=device, nb_element=1, normalize_input=True
                )

                # Denormalize pred_4
                pred_sat_4_denorm, pred_light_4_denorm = denormalize(
                    pred_4_norm[[0], :c_sat], pred_4_norm[[0], c_sat:], device
                )

                # Generate pred_5
                context_sat_gen2 = torch.cat([sample_batch["sat_patch_data"][[0], 1:4], pred_sat_4_denorm.permute(0, 2, 1, 3, 4)], dim=1)
                context_light_gen2 = torch.cat([sample_batch["lightning_patch_data"][[0], 1:4], pred_light_4_denorm.permute(0, 2, 1, 3, 4)], dim=1)
                batch_gen2 = {
                    "sat_patch_data": context_sat_gen2,
                    "lightning_patch_data": context_light_gen2,
                    "spatial_position": sample_batch["spatial_position"][[0], 1, :],
                }
                pred_5_norm, _ = full_image_generation(
                    forecast_model, batch_gen2, steps=gen_steps, device=device, nb_element=1, normalize_input=True
                )

                # Denormalize pred_5
                pred_sat_5_denorm, pred_light_5_denorm = denormalize(
                    pred_5_norm[[0], :c_sat], pred_5_norm[[0], c_sat:], device
                )

                # Prepare batch for correction (using first sample)
                dummy_sat_raw = sample_batch["sat_patch_data"][[0], 5:6]  # gt_6 as dummy? Wait, for input it's dummy, but for target it's gt_6
                dummy_light_raw = sample_batch["lightning_patch_data"][[0], 5:6]
                context_sat_raw = torch.cat([pred_sat_4_denorm, pred_sat_5_denorm], dim=2)
                context_light_raw = torch.cat([pred_light_4_denorm, pred_light_5_denorm], dim=2)


                batch_gen_corr = {
                    "sat_patch_data": torch.cat([context_sat_raw.permute(0, 2, 1, 3, 4), dummy_sat_raw], dim=1),
                    "lightning_patch_data": torch.cat([context_light_raw.permute(0, 2, 1, 3, 4), dummy_light_raw], dim=1),
                    "spatial_position": sample_batch["spatial_position"][[0], 1, :],  # position for frame 5
                }

                corrected_norm, target_norm = full_correction_generation(
                    correction_model, batch_gen_corr, steps=gen_steps, device=device, nb_element=1, normalize_input=True
                )

                # Denormalize for plotting
                corrected_sat, corrected_light = denormalize(corrected_norm[[0], :c_sat], corrected_norm[[0], c_sat:], device)
                target_sat, target_light = denormalize(target_norm[[0], :c_sat], target_norm[[0], c_sat:], device)
                pred5_sat, _ = denormalize(pred_5_norm[[0], :c_sat], pred_5_norm[[0], c_sat:], device)


                # Move to CPU and plot (simple example: plot first channel of sat)
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(pred5_sat[0, 11, 0].cpu().numpy(), cmap='gray')  # pred_5 channel 11
                axs[0].set_title('Predicted Frame 5')
                axs[1].imshow(corrected_sat[0, 11, 0].cpu().numpy(), cmap='gray')
                axs[1].set_title('Corrected Frame 6')
                axs[2].imshow(target_sat[0, 11, 0].cpu().numpy(), cmap='gray')
                axs[2].set_title('Ground Truth Frame 6')

                viz_dir = os.path.join(args.save_dir, "visualizations")
                os.makedirs(viz_dir, exist_ok=True)
                plt.savefig(os.path.join(viz_dir, f"correction_epoch_{epoch}.png"))
                plt.close()

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(correction_model)
                save_path = os.path.join(args.save_dir, f"correction_model_epoch_{epoch+1}.safetensors")
                os.makedirs(args.save_dir, exist_ok=True)
                save_file(unwrapped_model.state_dict(), save_path)
                print(f"Model saved to {save_path}")

        accelerator.wait_for_everyone()

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(correction_model)
        final_path = os.path.join(args.save_dir, "correction_model_final.safetensors")
        save_file(unwrapped_model.state_dict(), final_path)
        print(f"Training complete. Final model saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a correction model for weather forecasting using Accelerate.")
    parser.add_argument("--data_path", type=str, default="/workspace/dataset/", help="Path to the dataset.")
    parser.add_argument("--forecast_model_path", type=str, default="/workspace/meteolibre_model/models/models_world_shortcut/model_v1_mtg_world_lightning_shortcut_polynomial_e56.safetensors", help="Path to the pre-trained forecast model checkpoint.")
    parser.add_argument("--save_dir", type=str, default="./correction_models", help="Directory to save model checkpoints.")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
