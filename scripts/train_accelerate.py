"""
Training script for MeteoLibre using Hugging Face Accelerate.
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

# Add project root to sys.path
project_root = os.path.abspath("/workspace/meteolibre_model/")
sys.path.insert(0, project_root)


from meteolibre_model.dataset.dataset import MeteoLibreMapDataset
from meteolibre_model.diffusion.blurring_diffusion import trainer_step, full_image_generation
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
        localrepo="/workspace/data/data",  # Replace with your dataset path
        cache_size=8,
        seed=seed,
    )

    # Initialize DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1, #os.cpu_count() // 2,  # Use half the available CPUs
        pin_memory=True,
    )

    # Initialize model
    model = UNet_DCAE_3D(
        in_channels=12,  # Adjust based on your data
        out_channels=12,  # Adjust based on your data
        features=[32, 64, 128, 256],
        context_dim=4,
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

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_main_process,
        )
        for idx, batch in enumerate(progress_bar):

            # Perform training step
            with accelerator.accumulate(model):
                loss = trainer_step(model, batch, device)
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

                unwrapped_model = accelerator.unwrap_model(model)
                generated_images = full_image_generation(
                    unwrapped_model, x_context, device=accelerator.device
                )

                # Select one channel and one batch item for visualization
                generated_sample = generated_images[0, 0]  # Shape: (2, H, W)
                target_sample = x_target[0, 0].cpu()  # Shape: (2, H, W)

                all_frames = torch.cat([generated_sample, target_sample], dim=0)
                grid = make_grid(all_frames.unsqueeze(1), nrow=2)

                tb_tracker = accelerator.get_tracker("tensorboard")
                if tb_tracker:
                    tb_tracker.add_image("Generated vs Target", grid, epoch)

    # Save the model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(model.state_dict(), "meteolibre_model.pth")
        print("Training complete. Model saved to meteolibre_model.pth")


if __name__ == "__main__":
    main()
