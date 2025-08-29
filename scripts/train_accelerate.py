"""
Training script for MeteoLibre using Hugging Face Accelerate.
This script trains a rectified flow model using the MeteoLibreMapDataset and UNet_DCAE_3D.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration, LoggerType
from meteolibre_model.dataset.dataset import MeteoLibreMapDataset
from meteolibre_model.diffusion.diffusion import trainer_step
from meteolibre_model.models.dc_3dunet_film import UNet_DCAE_3D


def main():
    # Initialize Accelerator with bfloat16 precision and logging
    project_config = ProjectConfiguration(
        project_dir="runs/meteolibre_experiment",
        logging_dir="runs/meteolibre_experiment/logs",
    )

    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with=LoggerType.TENSORBOARD,
        project_config=project_config,
    )
    device = accelerator.device

    # Hyperparameters
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 10
    seed = 42
    gradient_clip_value = 1.0  # Gradient clipping value

    # Set seed for reproducibility
    set_seed(seed)

    # Initialize dataset
    dataset = MeteoLibreMapDataset(
        localrepo="path/to/your/dataset",  # Replace with your dataset path
        records_per_file=50,
        cache_size=8,
        seed=seed,
    )

    # Initialize DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count() // 2,  # Use half the available CPUs
        pin_memory=True,
    )

    # Initialize model
    model = UNet_DCAE_3D(
        in_channels=12,  # Adjust based on your data
        out_channels=12,  # Adjust based on your data
        features=[32, 64, 128, 256],
        context_dim=5,
        context_frames=4,
    )

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare for distributed training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for idx, batch in enumerate(dataloader):
            # Move batch to device
            batch_data = batch["patch_data"].to(device)

            # Perform training step
            with accelerator.accumulate(model):
                loss = trainer_step(model, batch_data)
                accelerator.backward(loss)

                # Gradient clipping
                accelerator.clip_grad_norm_(model.parameters(), gradient_clip_value)

                optimizer.step()
                optimizer.zero_grad()

                accelerator.log(
                    {"Loss/train_trained": loss.item()},
                    step=epoch * len(dataloader) + idx,
                )

                total_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)

        # Log to Accelerate
        accelerator.log({"Loss/train": avg_loss}, step=epoch)

        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save the model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(model.state_dict(), "meteolibre_model.pth")
        print("Training complete. Model saved to meteolibre_model.pth")


if __name__ == "__main__":
    main()
