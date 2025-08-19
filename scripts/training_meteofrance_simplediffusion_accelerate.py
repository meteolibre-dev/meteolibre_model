"""
Script for module training using accelerate
"""
import sys
import os
import random

# Add project root to sys.path
project_root = os.path.abspath("/workspace/meteolibre_model/")
sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from heavyball import ForeachSOAP

from accelerate import Accelerator
from safetensors.torch import save_file, load_file
from huggingface_hub import HfApi
from tqdm import tqdm
import os
import torchvision
import einops

from meteolibre_model.datasets.dataset_meteofrance_v3 import MeteoLibreDatasetHF
from meteolibre_model.dit.model_meteofrance_simplediffusion import Simple3DDiffusionModel

# Configuration
#PATHDATA = ["/workspace/data/hf_dataset"]
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
NUM_WORKERS = 16
NUM_EPOCHS = 100
GRADIENT_ACCUMULATION_STEPS = 8
GRADIENT_CLIP_VAL = 1.0
LOG_EVERY_N_STEPS = 20
SAVE_EVERY_N_EPOCHS = 5
STEPS_PER_EPOCH = 5900  # Define steps per epoch for IterableDataset
MODEL_DIR = "models/meteolibre_simplediffusion_accelerate/"
IMAGE_LOG_DIR = os.path.join(MODEL_DIR, "images")

def log_sample_image(model, batch, step, accelerator):
    os.makedirs(IMAGE_LOG_DIR, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        x_image, _, _, _ = model.prepare_target(batch, accelerator.device)
        input_meteo_frames = x_image[:, :model.nb_back]

        
        x_hour = batch["hour"].clone().detach().float().unsqueeze(1)
        x_minute = batch["minute"].clone().detach().float().unsqueeze(1)
        
        sample = model.sample(input_meteo_frames, x_hour, x_minute)
        
        # Assuming the radar channel is at index 5
        sample_radar = sample[0, :, 5, :, :]
        sample_radar = einops.rearrange(sample_radar, 't h w -> t 1 h w')
        
        save_path = os.path.join(IMAGE_LOG_DIR, f"sample_radar_step_{step}.png")
        torchvision.utils.save_image(sample_radar, save_path, normalize=True, value_range=(-2., 2.))
        accelerator.print(f"Saved sample image to {save_path}")
        
        # Assuming the sat channel is at index -1
        sample_sat = sample[0, :, -1, :, :]
        sample_sat = einops.rearrange(sample_sat, 't h w -> t 1 h w')
        
        save_path = os.path.join(IMAGE_LOG_DIR, f"sample_sat_step_{step}.png")
        torchvision.utils.save_image(sample_sat, save_path, normalize=True, value_range=(-2., 2.))
        accelerator.print(f"Saved sample image to {save_path}")
        
        # Assuming the lancover channel is at index 0
        sample_landcover = sample[0, :, 0, :, :]
        sample_landcover = einops.rearrange(sample_landcover, 't h w -> t 1 h w')
        
        save_path = os.path.join(IMAGE_LOG_DIR, f"sample_landcover_step_{step}.png")
        torchvision.utils.save_image(sample_landcover, save_path, normalize=True, value_range=(-2., 2.))
        accelerator.print(f"Saved sample image to {save_path}")

        # Log target images for comparison
        target_frames = x_image[0, model.nb_back:(model.nb_back + model.nb_future)]

        # Assuming the radar channel is at index 5
        target_radar = target_frames[:, 5, :, :]
        target_radar = einops.rearrange(target_radar, 't h w -> t 1 h w')
        save_path = os.path.join(IMAGE_LOG_DIR, f"target_radar_step_{step}.png")
        torchvision.utils.save_image(target_radar, save_path, normalize=True, value_range=(-2., 2.))
        accelerator.print(f"Saved target radar image to {save_path}")

        # Assuming the sat channel is at index -1
        target_sat = target_frames[:, -1, :, :]
        target_sat = einops.rearrange(target_sat, 't h w -> t 1 h w')
        save_path = os.path.join(IMAGE_LOG_DIR, f"target_sat_step_{step}.png")
        torchvision.utils.save_image(target_sat, save_path, normalize=True, value_range=(-2., 2.))
        accelerator.print(f"Saved target sat image to {save_path}")

        # Assuming the landcover channel is at index 0
        target_landcover = target_frames[:, 0, :, :]
        target_landcover = einops.rearrange(target_landcover, 't h w -> t 1 h w')
        save_path = os.path.join(IMAGE_LOG_DIR, f"target_landcover_step_{step}.png")
        torchvision.utils.save_image(target_landcover, save_path, normalize=True, value_range=(-2., 2.))
        accelerator.print(f"Saved target landcover image to {save_path}")
    model.train()

def main():
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        log_with="tensorboard", project_dir="."
    )

    hps = {"batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE}
    accelerator.init_trackers("tb_simplediffusion_" +  str(random.randint(0, 1000)), config=hps)

    # Initialize Dataset and DataLoader
    dataset = MeteoLibreDatasetHF(localrepo="/workspace/data")
    train_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = Simple3DDiffusionModel(
        parametrization="velocity",
        schedule="shifted_cosine",
    )
    
    # load model
    #PATH = "/workspace/meteolibre_model/models/meteolibre_simplediffusion_accelerate/epoch_30.safetensors"
    #weights = load_file(PATH, device="cuda:0")
    #model.load_state_dict(weights)

    # Initialize Optimizer
    #optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer = ForeachSOAP(model.parameters(), lr=LEARNING_RATE, warmup_steps=100)

    # Prepare for training with accelerate
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Training Loop
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        pbar = tqdm(train_dataloader, total=STEPS_PER_EPOCH, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for step, batch in enumerate(pbar):
            # Manually break after STEPS_PER_EPOCH
            if step >= STEPS_PER_EPOCH:
                break
            
            if step == 0:
                # log image at the epoch end
                log_sample_image(model, batch, global_step, accelerator)

            with accelerator.accumulate(model):
                losses = model.compute_loss(batch, accelerator.device)
                loss = losses["total_loss"]
                
                if torch.isnan(loss):
                    accelerator.print(f"NaN loss detected at epoch {epoch}, step {step}. Skipping.")
                    continue

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
                optimizer.step()
                optimizer.zero_grad()

            if (global_step + 1) % LOG_EVERY_N_STEPS == 0:
                pbar.set_postfix(loss=loss.item())
                accelerator.print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{step+1}/{STEPS_PER_EPOCH}], Loss: {loss.item():.4f}")
                for loss_name, loss_value in losses.items():
                    accelerator.log({loss_name: loss_value.item()}, step=global_step)
            
            global_step += 1
        

            

        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = f"{MODEL_DIR}epoch_{epoch+1}.safetensors"
            save_file(unwrapped_model.state_dict(), save_path)
            accelerator.print(f"Model saved to {save_path}")
            
            

    accelerator.end_training()
    print("Training finished!")

    # Save final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    

    # final_save_path = "diffusion_pytorch_model.safetensors"
    # save_file(unwrapped_model.state_dict(), final_save_path)

    # # Upload to Hugging Face Hub
    # try:
    #     api = HfApi()
    #     api.upload_file(
    #         path_or_fileobj=final_save_path,
    #         path_in_repo="weights_vae/model_meteofrance_rope_vae.safetensors",
    #         repo_id="Forbu14/meteolibre",
    #         repo_type="model",
    #     )
    #     print("Model successfully uploaded to Hugging Face Hub.")
    # except Exception as e:
    #     print(f"Failed to upload model to Hugging Face Hub: {e}")

if __name__ == "__main__":
    main()
