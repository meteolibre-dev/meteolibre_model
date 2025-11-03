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

from heavyball import ForeachSOAP

# Add project root to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from meteolibre_model.dataset.dataset_for_diffusion import MeteoFranceDataset
from meteolibre_model.model.unet_model import Unet
from meteolibre_model.diffusion.rectified_flow_lightning_shortcut import (
    full_image_generation,
    normalize,
    denormalize,
)
from meteolibre_model.diffusion.rectified_flow_correction_training import regression_trainer_step
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
        "learning_rate": learning_rate,
        "gen_steps": gen_steps,
    }

    # Set seed for reproducibility
    set_seed(seed)

    accelerator.init_trackers(
        "correction_model_training_rectified_flow_" + id_run, config=hps
    )

    # Initialize dataset
    dataset = MeteoFranceDataset(
        data_path=args.data_path,
        split="train",
        sequence_length=7,
        use_lightning=True
    )

    # Initialize DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Load pre-trained forecast model
    model_params = params["model"]
    forecast_model = DualUNet3DFiLM(**model_params)

    if not os.path.exists(args.forecast_model_path):
        raise FileNotFoundError(f"Forecast model checkpoint not found at {args.forecast_model_path}")
    checkpoint = torch.load(args.forecast_model_path, map_location=device)
    forecast_model.load_state_dict(checkpoint)
    forecast_model.to(device)
    forecast_model.eval()
    print(f"Loaded forecast model from {args.forecast_model_path}")

    # Initialize correction model
    model_params_corr = params["model_correction"]
    correction_model = DualUNet3DFiLM(**model_params_corr)

    # Initialize optimizer
    optimizer = ForeachSOAP(correction_model.parameters(), lr=learning_rate, foreach=False)

    # Prepare for distributed training (only correction_model and optimizer)
    correction_model, forecast_model, optimizer, dataloader = accelerator.prepare(forecast_model, correction_model, optimizer, dataloader)

    global_step = 0

    # Training loop
    for epoch in range(num_epochs):
        correction_model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_main_process,
        )
        for batch in progress_bar:
            with accelerator.accumulate(correction_model):
                # Perform training step using the correction logic
                loss = regression_trainer_step(
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
                # This would require adapting full_image_generation for visualization
                # For now, skip detailed viz, or implement simple logging
                pass

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
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--forecast_model_path", type=str, required=True, help="Path to the pre-trained forecast model checkpoint.")
    parser.add_argument("--save_dir", type=str, default="./correction_models", help="Directory to save model checkpoints.")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
