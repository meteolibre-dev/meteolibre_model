"""
Training script for MeteoLibre using a mixed strategy of standard and autoregressive training.

This script combines two training approaches for a rectified flow model:
1. The standard `trainer_step` using 5-frame sequences.
2. The `trainer_step_autoregressive` using 7-frame sequences to train the model on its own outputs.

At each training step, one of the two methods is chosen with a 50% probability.
"""

import sys
import os
import torch
import random
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from datetime import datetime
import yaml
from itertools import cycle

from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import save_file
from safetensors.torch import load_file

# custom optimizer
from heavyball import ForeachSOAP

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from meteolibre_model.dataset.dataset_mtg_lightning import MeteoLibreMapDataset
from meteolibre_model.dataset.dataset_mtg_lightning_7frames import MeteoLibreMapDataset7Frames
from meteolibre_model.diffusion.rectified_flow_lightning_shortcut import (
    trainer_step,
    full_image_generation,
)
from meteolibre_model.diffusion.rectified_flow_lightning_shortcut_autoregressive import (
    trainer_step_autoregressive,
)

from meteolibre_model.models.unet3d_film_dual import DualUNet3DFiLM

# Load config
config_path = os.path.join(project_root, "meteolibre_model/config/configs.yml")
with open(config_path) as f:
    config = yaml.safe_load(f)
params = config['model_v1_mtg_world_lightning_shortcut']

def main():
    # Initialize Accelerator
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=".",
        kwargs_handlers=[kwargs],
    )
    device = accelerator.device

    # Load hyperparameters
    LOG_EVERY_N_STEPS = params['log_every_n_steps']
    SAVE_EVERY_N_EPOCHS = params['save_every_n_epochs']
    MODEL_DIR = params['model_dir']
    PARAMETRIZATION = params['parametrization']
    INTERPOLATION = params.get('interpolation', 'linear')
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    num_epochs = params['num_epochs']
    seed = params['seed']
    sigma_noise_input = params['sigma_noise_input']
    gradient_clip_value = params['gradient_clip_value']
    id_run = str(datetime.utcnow())[:19]
    set_seed(seed)

    hps = {"batch_size": batch_size, "learning_rate": learning_rate, "mixed_training": 0.5}
    accelerator.init_trackers(
        "lightning_shortcut-autoregressive-mixed-training-" + id_run, config=hps
    )

    # Initialize datasets
    dataset_5_frames = MeteoLibreMapDataset(
        localrepo=params['dataset_path'], cache_size=4, seed=seed, nb_temporal=5
    )
    dataset_7_frames = MeteoLibreMapDataset7Frames(
        localrepo=params['dataset_path'], cache_size=4, seed=seed + 1, nb_temporal=7
    )

    # Initialize DataLoaders
    dataloader_5_frames = DataLoader(
        dataset_5_frames, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )
    dataloader_7_frames = DataLoader(
        dataset_7_frames, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    # Initialize model
    model_params = params["model"]
    model = DualUNet3DFiLM(**model_params)

    print("loading model")
    model_path = "/workspace/meteolibre_model/models/models_world_shortcut/model_v1_mtg_world_lightning_shortcut_polynomial_e56.safetensors"
    state_dict = load_file(model_path)

    model.load_state_dict(state_dict)

    learning_rate = 0.0001

    # Initialize optimizer
    # optimizer = ForeachSOAP(model.parameters(), lr=learning_rate, foreach=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Prepare for distributed training
    model, optimizer, dataloader_5, dataloader_7 = accelerator.prepare(
        model, optimizer, dataloader_5_frames, dataloader_7_frames
    )

    global_step = 0
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Use the shorter dataloader to define the number of steps per epoch
        progress_bar = tqdm(
            zip(dataloader_5, dataloader_7),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=min(len(dataloader_5), len(dataloader_7)),
            disable=not accelerator.is_main_process,
        )

        for batch5, batch7 in progress_bar:
            with accelerator.accumulate(model):
                # Randomly choose between standard and autoregressive training
                if random.random() < 0.5:
                    loss, loss_sat, loss_kpi = trainer_step(
                        model, batch5, device, parametrization=PARAMETRIZATION, interpolation=INTERPOLATION, sigma=sigma_noise_input
                    )
                else:
                    loss, loss_sat, loss_kpi = trainer_step_autoregressive(
                        model, batch7, device, parametrization=PARAMETRIZATION, interpolation=INTERPOLATION, sigma=sigma_noise_input
                    )
                
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), gradient_clip_value)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % LOG_EVERY_N_STEPS == 0 and accelerator.is_main_process:
                    accelerator.log({"Loss/train_step": loss.item()}, step=global_step)
                    accelerator.log({"Loss_sat/train_step": loss_sat.item()}, step=global_step)
                    accelerator.log({"Loss_kpi/train_step": loss_kpi.item()}, step=global_step)

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / min(len(dataloader_5), len(dataloader_7))
        accelerator.log({"Loss/train_epoch": avg_loss}, step=epoch)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if accelerator.is_main_process:
            with torch.no_grad():
                unwrapped_model = accelerator.unwrap_model(model)
                # Use a batch from the 5-frame loader for consistent visualization
                vis_batch = next(iter(dataloader_5))
                generated_images, x_target = full_image_generation(
                    unwrapped_model, vis_batch, device=accelerator.device, parametrization=PARAMETRIZATION
                )

                generated_sample = generated_images[0, 11]
                target_sample = x_target[0, 11].cpu()
                all_frames = torch.cat([generated_sample, target_sample], dim=0) / 8.0
                grid = make_grid(all_frames.clamp(-10, 10).unsqueeze(1), nrow=2)
                
                tb_tracker = accelerator.get_tracker("tensorboard")
                if tb_tracker:
                    tb_tracker.writer.add_image("Generated vs Target", grid, epoch)

        if (epoch) % SAVE_EVERY_N_EPOCHS == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = f"{MODEL_DIR}epoch_{epoch + 1}_autoregressive.safetensors"
                os.makedirs(MODEL_DIR, exist_ok=True)
                save_file(unwrapped_model.state_dict(), save_path)
                accelerator.print(f"Model saved to {save_path}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        final_save_path = "meteolibre_model_autoregressive_final.safetensors"
        save_file(unwrapped_model.state_dict(), final_save_path)
        print(f"Training complete. Model saved to {final_save_path}")

if __name__ == "__main__":
    main()
