"""
Script for module training using accelerate
"""
import sys
import os
sys.path.insert(0, "/teamspace/studios/this_studio/meteolibre_model/")


import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from heavyball import ForeachSOAP

from accelerate import Accelerator
from safetensors.torch import save_file
from huggingface_hub import HfApi
from tqdm import tqdm
import os
import torchvision
import einops

from meteolibre_model.datasets.dataset_meteofrance_v2 import MeteoLibreDataset
from meteolibre_model.dit.model_meteofrance_simplediffusion import Simple3DDiffusionModel

# Configuration
#PATHDATA = ["/workspace/data/hf_dataset_v0/", "/workspace/data/hf_dataset_v1/"]
PATHDATA = ["/teamspace/studios/this_studio/data/hf_dataset/"]
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
NUM_WORKERS = 20
NUM_EPOCHS = 100
GRADIENT_ACCUMULATION_STEPS = 2
GRADIENT_CLIP_VAL = 1.0
LOG_EVERY_N_STEPS = 5
SAVE_EVERY_N_EPOCHS = 1
MODEL_DIR = "models/meteolibre_simplediffusion_accelerate/"
IMAGE_LOG_DIR = os.path.join(MODEL_DIR, "images")

def log_sample_image(model, batch, step, accelerator):
    os.makedirs(IMAGE_LOG_DIR, exist_ok=True)
    
    with torch.no_grad():
        x_image, _, _, _ = model.prepare_target(batch, accelerator.device)
        input_meteo_frames = x_image[:, :model.nb_back]
        
        x_hour = batch["hour"].clone().detach().float().unsqueeze(1)
        x_minute = batch["minute"].clone().detach().float().unsqueeze(1)
        
        sample = model.sample(input_meteo_frames, x_hour, x_minute)
        
        # Assuming the radar channel is at index 5
        sample_radar = sample[0, :, 5, :, :]
        sample_radar = einops.rearrange(sample_radar, 't h w -> t 1 h w')
        
        save_path = os.path.join(IMAGE_LOG_DIR, f"sample_step_{step}.png")
        torchvision.utils.save_image(sample_radar, save_path, normalize=True)
        accelerator.print(f"Saved sample image to {save_path}")

def main():
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        log_with="tensorboard", project_dir="."
    )

    hps = {"batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE}
    accelerator.init_trackers("tb_simplediffusion", config=hps)

    # Initialize Dataset and DataLoader
    dataset = MeteoLibreDataset(directory=PATHDATA)
    train_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = Simple3DDiffusionModel(
        parametrization="velocity",
        schedule="shifted_cosine",
    )

    # Initialize Optimizer
    #optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer = ForeachSOAP(model.parameters(), lr=LEARNING_RATE, warmup_steps=100)

    # Prepare for training with accelerate
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
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

            global_step = epoch * len(train_dataloader) + step
            if (global_step + 1) % LOG_EVERY_N_STEPS == 0:
                accelerator.print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
                for loss_name, loss_value in losses.items():
                    accelerator.log({loss_name: loss_value.item()}, step=global_step)
                
        # log image at the epoch end
        log_sample_image(model, batch, global_step, accelerator)
            

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
