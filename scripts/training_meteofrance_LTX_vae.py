"""
Script for module training
"""

from safetensors.torch import save_file
from huggingface_hub import HfApi

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger #WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader

from meteolibre_model.datasets.dataset_meteofrance_v2 import (
    MeteoLibreDataset,
)

from meteolibre_model.vae.pl_model_meteofrance_LTX_vae import VAEMeteoLibrePLModelLTXVae


import torch

torch.set_float32_matmul_precision("medium")

import torch._dynamo
torch._dynamo.config.suppress_errors = True

PATHDATA = "/workspace/data/hf_dataset/"

def init_dataset():
    dataset = MeteoLibreDataset(directory=PATHDATA)

    return dataset


if __name__ == "__main__":
    dataset = init_dataset()

    # For simplicity, use the same dataset for training and validation.
    # In a real scenario, you should have separate datasets.
    train_dataset = dataset
    val_dataset = dataset  # Using same dataset for now

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=16,
    )  # )
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=True
    )  # Optional, if you want validation

    model = VAEMeteoLibrePLModelLTXVae(
        test_dataloader=val_dataloader,
        dir_save="./",
    )

    #model.compile()

    callback = ModelCheckpoint(
        every_n_epochs=1,
        save_last=True,
        dirpath="models/meteolibre_vae_ltx/",
        save_weights_only=True,
    )

    logger = TensorBoardLogger("tb_logs/", name="ltx_vae")
    #logger = WandbLogger(project="meteolibre_meteofrance_model_vae")

    trainer = pl.Trainer(
        max_time={"hours": 4},
        logger=logger,
        #accumulate_grad_batches=2,
        #fast_dev_run=True,
        # accelerator="cpu", # debug
        callbacks=[callback],
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        enable_checkpointing=True,
    )  # fast_dev_run=True for quick debugging

    trainer.fit(
        model, train_dataloader, val_dataloader
    )  # Pass val_dataloader if you have validation step in model

    print("Training finished!")

    # Save the model in safetensors format
    save_file(model.model.state_dict(), "diffusion_pytorch_model.safetensors")

    # torch.save(model.model.state_dict(), "model_vae.pt")

    # push file to hub
    api = HfApi()
    api.upload_file(
        path_or_fileobj="diffusion_pytorch_model.safetensors",
        path_in_repo="weights_vae/model_meteofrance_rope_vae.safetensors",
        repo_id="Forbu14/meteolibre",
        repo_type="model",
    )
