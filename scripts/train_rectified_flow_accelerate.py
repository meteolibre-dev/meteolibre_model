"""
Training script for MeteoLibre using Hugging Face Accelerate with Rectified Flow.
This script trains a rectified flow model using the MeteoLibreMapDataset and UNet_DCAE_3D.
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration, LoggerType
from tqdm.auto import tqdm
import random

from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import save_file, load_file

# Add project root to sys.path
project_root = os.path.abspath("/workspace/meteolibre_model/")
sys.path.insert(0, project_root)


from meteolibre_model.dataset.dataset import MeteoLibreMapDataset
from meteolibre_model.diffusion.rectified_flow import (
    trainer_step,
    full_image_generation,
    normalize,
)
from meteolibre_model.models.dc_3dunet_film import UNet_DCAE_3D


def main():
    # Initialize Accelerator with bfloat16 precision and logging
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=".",
        kwargs_handlers=[kwargs],
    )
    device = accelerator.device

    # Hyperparameters
    LOG_EVERY_N_STEPS = 5
    SAVE_EVERY_N_EPOCHS = 5
    MODEL_DIR = "models/"
    PARAMETRIZATION = "residual"
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 200
    seed = 42
    gradient_clip_value = 1.0  # Gradient clipping value
    id_run = str(random.randint(0, 1000))
    # Set seed for reproducibility
    set_seed(seed)

    hps = {"batch_size": batch_size, "learning_rate": learning_rate}

    accelerator.init_trackers(
        "meteofrance-eps-prediction-training-rectified-flow_" + id_run, config=hps
    )

    # Initialize dataset
    dataset = MeteoLibreMapDataset(
        localrepo="/workspace/data/data",  # Replace with your dataset path
        cache_size=8,
        seed=seed,
    )

    # Initialize DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # os.cpu_count() // 2,  # Use half the available CPUs
        pin_memory=True,
    )

    # Initialize model
    model = UNet_DCAE_3D(
        in_channels=12,  # Adjust based on your data
        out_channels=12,  # Adjust based on your data
        features=[64, 128, 256],
        context_dim=4,
        context_frames=4,
        num_additional_resnet_blocks=1
    )

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare for distributed training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    global_step = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_main_process,
        )
        for batch in progress_bar:
            # Perform training step
            with accelerator.accumulate(model):
                loss = trainer_step(model, batch, device, PARAMETRIZATION)
                accelerator.backward(loss)

                # Gradient clipping
                accelerator.clip_grad_norm_(model.parameters(), gradient_clip_value)

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % LOG_EVERY_N_STEPS == 0:
                    if accelerator.is_main_process:
                        accelerator.log(
                            {"Loss/train_trained": loss.item()},
                            step=global_step,
                        )

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)

        # Log to Accelerate
        accelerator.log({"Loss/train": avg_loss}, step=epoch)

        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if accelerator.is_main_process:
            with torch.no_grad():
                permuted_batch_data = batch["patch_data"].permute(0, 2, 1, 3, 4)
                x_context = permuted_batch_data[:, :, :4]
                x_target = permuted_batch_data[:, :, 4:]

                x_target = normalize(x_target, device)

                unwrapped_model = accelerator.unwrap_model(model)
                generated_images = full_image_generation(
                    unwrapped_model, batch, x_context, device=accelerator.device, parametrization=PARAMETRIZATION
                )

                # Select one channel and one batch item for visualization
                generated_sample = generated_images[0, -1]  # Shape: (2, H, W)
                target_sample = x_target[0, -1].cpu()  # Shape: (2, H, W)

                all_frames = torch.cat([generated_sample, target_sample], dim=0) / 8.
                all_frames = all_frames.clamp(-10, 10)

                grid = make_grid(all_frames.unsqueeze(1), nrow=2)
                grid_normalized = make_grid(
                    (all_frames.unsqueeze(1) - all_frames.min())
                    / (all_frames.max() - all_frames.min()),
                    nrow=2,
                )

                tb_tracker = accelerator.get_tracker("tensorboard")
                if tb_tracker:
                    tb_tracker.writer.add_image("Generated vs Target", grid, epoch)
                    tb_tracker.writer.add_image("Generated vs Target (normalized)", grid_normalized, epoch)


        # This part for saving the model was already correct
        if (epoch) % SAVE_EVERY_N_EPOCHS == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                # Save the EMA model's state dictionary
                save_path = f"{MODEL_DIR}epoch_{epoch + 1}_rectified_flow.safetensors"
                os.makedirs(MODEL_DIR, exist_ok=True)
                save_file(unwrapped_model.state_dict(), save_path)
                accelerator.print(f"Model saved to {save_path}")

        accelerator.wait_for_everyone()

    # Save the model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(model.state_dict(), "meteolibre_model_rectified_flow.pth")
        print("Training complete. Model saved to meteolibre_model_rectified_flow.pth")


if __name__ == "__main__":
    main()