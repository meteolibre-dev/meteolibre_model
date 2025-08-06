"""
In this module we will test a new architecture for the VAE.
IMG -> Encoder (CNN + DiT) -> Latent Space -> Decoder (DiT + CNN) -> IMG
"""

import os
import glob

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

import einops

from torch.optim import optimizer

import wandb
from PIL import Image

from heavyball import ForeachSOAP

from diffusers import AutoencoderKLLTXVideo


class VAEMeteoLibrePLModelLTXVae(pl.LightningModule):
    """
    PyTorch Lightning module for the MeteoLibre model.

    VAE autoencoder
    """

    def __init__(
        self,
        learning_rate=1e-3,
        test_dataloader=None,
        dir_save="../",
        input_channels=13,
        output_channels=13,
        latent_dim=128,
        coefficient_reg=0.000001,
        nb_frames=5,
    ):
        """
        Initialize the MeteoLibrePLModel.

        Args:
            TODO later
        """
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.latent_dim = latent_dim
        self.coefficient_reg = coefficient_reg
        self.learning_rate = learning_rate
        self.patch_size = 2
        self.nb_frames = nb_frames

        self.model = AutoencoderKLLTXVideo(
            in_channels=input_channels,
            out_channels=output_channels,
            latent_channels=latent_dim,
            block_out_channels=(128 // 2, 256 // 2, 512 // 2, 512 // 2),
            down_block_types=(
                "LTXVideoDownBlock3D",
                "LTXVideoDownBlock3D",
                "LTXVideoDownBlock3D",
                "LTXVideoDownBlock3D",
            ),
            decoder_block_out_channels=(128 // 2, 256 // 2, 512 // 2, 512 // 2),
            layers_per_block=(4, 3, 3, 3, 4),
            patch_size=4,
            spatial_compression_ratio=8,
            decoder_layers_per_block=(4, 3, 3, 3, 4),
            spatio_temporal_scaling=(True, True, False, False),
            decoder_spatio_temporal_scaling=(True, True, False, False),
            downsample_type=("conv", "conv", "conv", "conv"),
            upsample_factor=(1, 1, 1, 1),
            decoder_inject_noise=(False, False, False, False, False),
            upsample_residual=(False, False, False, False),
        )

        # self.final_layer = nn.Linear(latent_dim*2, 2 * 2 * latent_dim, bias=True)
        self.learning_rate = learning_rate
        self.test_dataloader = test_dataloader

        self.dir_save = dir_save

    def encode(self, x_image):
        """
        Forward pass through the model.

        Args:
            x_image (torch.Tensor): Input image tensor of size (b, nb_frame, c, h, w)

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        batch_size, nb_frame, c, h, w = x_image.shape

        x_image = einops.rearrange(
            x_image, "batch_size nb_frame c h w -> batch_size c nb_frame h w"
        )

        encode_input = self.model.encode(x_image)

        return (
            encode_input.latent_dist.mean,
            encode_input.latent_dist.logvar,
            encode_input.latent_dist.sample(),
        )

    def decode(self, z):
        # decoder
        final_image = self.model.decode(z).sample

        # reshape
        final_image = einops.rearrange(
            final_image, "b c t h w -> b t c h w", t=self.nb_frames
        )

        return final_image

    def forward(self, x_image):
        """P
        Forward pass through the model.

        Args:
            x_image (torch.Tensor): Input image tensor of size (b, nb_frame, c, h, w)

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        final_latent_mean, final_latent_logvar, sample = self.encode(x_image)

        final_image = self.decode(sample)

        return final_image, (final_latent_mean, final_latent_logvar, sample)

    def training_step(self, batch, batch_idx):
        """
        Training step for the PyTorch Lightning module.
        """

        radar_data = batch["radar"].unsqueeze(-1)[:, :self.nb_frames]
        groundstation_data = batch["groundstation"][:, :self.nb_frames]

        groundheight = batch["ground_height"].unsqueeze(-1).unsqueeze(1).float()
        landcover = batch["landcover"].unsqueeze(1).float()

        groundheight = groundheight.repeat(1, self.nb_frames, 1, 1, 1)
        landcover = landcover.repeat(1, self.nb_frames, 1, 1, 1)
        
        # little correction
        groundstation_data = torch.where(
            groundstation_data == -100, -4, groundstation_data
        )

        # mask radar
        mask_radar = torch.ones_like(radar_data).bool()
        mask_groundstation = groundstation_data != -4

        # random masking to force generalization
        groundstation_data_corrupt = torch.where(
            torch.rand_like(groundstation_data) > 0.3, groundstation_data, -4
        )

        # concat the two elements
        x_image = torch.cat((groundheight, landcover, radar_data, groundstation_data), dim=-1)
        x_image = x_image.permute(0, 1, 4, 2, 3)  # (N, nb_frame, C, H, W)

        x_image_corrupt = torch.cat((groundheight, landcover, radar_data, groundstation_data_corrupt), dim=-1)
        x_image_corrupt = x_image_corrupt.permute(0, 1, 4, 2, 3)  # (N, nb_frame, C, H, W)

        mask_radar = mask_radar.permute(0, 1, 4, 2, 3)
        mask_groundstation = mask_groundstation.permute(0, 1, 4, 2, 3)
        
        # forward pass
        final_image, (final_latent_mean, final_latent_logvar, z) = self(x_image_corrupt)

        reconstruction_loss_radar = F.mse_loss(final_image[:, :, [5], :, :][mask_radar], x_image[:, :, [5], :, :][mask_radar], reduction="mean")
        reconstruction_loss_groundstation = F.mse_loss(final_image[:, :, 6:, :, :][mask_groundstation], x_image[:, :, 6:, :, :][mask_groundstation], reduction="mean")
        reconstruction_ground = F.mse_loss(final_image[:, :, :5, :, :], x_image[:, :, :5, :, :], reduction="mean")
        
        reconstruction_loss = (
            reconstruction_loss_radar + reconstruction_loss_groundstation * 0.3 + reconstruction_ground * 0.01
        )

        # logging data
        self.log("reconstruction_loss", reconstruction_loss)
        self.log("reconstruction_loss_radar", reconstruction_loss_radar)
        self.log("reconstruction_loss_groundstation", reconstruction_loss_groundstation)
        self.log("reconstruction_loss_ground_info", reconstruction_ground)

        # KL regularizatio loss
        kl_loss = torch.mean(
            -0.5
            * torch.sum(
                1
                + final_latent_logvar
                - final_latent_mean**2
                - final_latent_logvar.exp(),
                dim=1,
            ),
        )

        self.log("kl_loss", kl_loss)

        loss = reconstruction_loss + self.coefficient_reg * kl_loss

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        optimizer = ForeachSOAP(self.parameters(), lr=self.learning_rate, foreach=False, warmup_steps=50)
        return optimizer

    @torch.no_grad()
    def generate_one(self, nb_batch=1, nb_step=100):
        # Assuming batch is a dictionary returned by TFDataset
        # generate a random (nb_batch, 1, 256, 256) tensor
        for batch in self.test_dataloader:
            break

        # convert everything to device
        for key in batch.keys():
            batch[key] = batch[key].to(self.device)

        radar_data = batch["radar"].unsqueeze(-1)[:, :self.nb_frames]
        groundstation_data = batch["groundstation"][:, :self.nb_frames]

        groundheight = batch["ground_height"].unsqueeze(-1).unsqueeze(1).float()
        landcover = batch["landcover"].unsqueeze(1).float()

        groundheight = groundheight.repeat(1, self.nb_frames, 1, 1, 1)
        landcover = landcover.repeat(1, self.nb_frames, 1, 1, 1)

        # little correction
        groundstation_data = torch.where(
            groundstation_data == -100, -4, groundstation_data
        )

        # mask radar
        mask_radar = torch.ones_like(radar_data)
        mask_groundstation = groundstation_data != -4

        # concat the two elements
        x_image = torch.cat((groundheight, landcover, radar_data, groundstation_data), dim=-1)

        x_image = x_image.permute(0, 1, 4, 2, 3)  # (N, nb_frame, C, H, W)

        # forward pass
        final_image, latent = self(x_image)

        return final_image, x_image, latent

    # on epoch end of training
    def on_train_epoch_end(self):
        # generate image
        self.eval()

        with torch.no_grad():
            result, x_image_future, latent = self.generate_one(nb_batch=1, nb_step=100)

            print("latent mean", latent[2].mean())
            print("latent std", latent[2].std())

            for i in range(result.shape[2]):
                self.save_image(result[0, 0, i, :, :], name_append=f"result_T_{i}")
                self.save_image(x_image_future[0, 0, i, :, :], name_append=f"target_T_{i}")

            # self.save_gif(result, name_append="result_gif_vae")
            # self.save_gif(x_image_future, name_append="target_gif_vae")

        self.train()

    def save_image(self, result, name_append="result"):
        radar_image = result.cpu().detach().numpy()  # (H, W)

        fname = self.dir_save + f"data/{name_append}_epoch_{self.current_epoch}.png"

        plt.figure(figsize=(20, 20))
        plt.imshow(radar_image, vmin=-1, vmax=2)
        plt.colorbar()

        plt.savefig(fname, bbox_inches="tight", pad_inches=0)
        plt.close()

        img = plt.imread(fname)[:, :, :3]
        img = img.transpose((2, 0, 1))

        # logging image into wandb
        self.logger.experiment.add_image(
                name_append, img, global_step=self.global_step
            )

    def save_gif(self, result, name_append="result", duration=10):
        nb_frame = result.shape[1]

        file_name_list = []

        for i in range(nb_frame):
            fname = (
                self.dir_save
                + f"data/{name_append}_radar_epoch_{self.current_epoch}_{i}.png"
            )

            radar_image = result[0, i, 0, :, :].cpu().numpy()

            plt.figure(figsize=(20, 20))
            plt.imshow(radar_image, vmin=-4, vmax=2)
            plt.colorbar()

            plt.savefig(fname, bbox_inches="tight", pad_inches=0)
            plt.close()

            file_name_list.append(fname)

        create_gif_pillow(
            image_paths=file_name_list,
            output_path=self.dir_save
            + f"data/{name_append}_radar_epoch_{self.current_epoch}.gif",
            duration=duration,
        )


def create_gif_pillow(image_paths, output_path, duration=100):
    """
    Creates a GIF from a list of image paths using Pillow.

    Args:
      image_paths: A list of strings, where each string is the path to an image file.
      output_path: The path where the GIF will be saved (e.g., 'output.gif').
      duration: The duration (in milliseconds) to display each frame in the GIF.
    """
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
        except FileNotFoundError:
            print(f"Error: Image not found at {path}")
            return

    if images:
        first_frame = images[0]
        remaining_frames = images[1:]

        first_frame.save(
            output_path,
            save_all=True,
            append_images=remaining_frames,
            duration=duration,
            loop=0,  # 0 means loop indefinitely
        )
        print(f"GIF created successfully at {output_path}")

    # remove the png images after creating the gif
    for path in image_paths:
        os.remove(path)

    else:
        print("No valid images found to create GIF.")
