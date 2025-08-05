"""
meteolibre_model/meteolibre_model/pl_model.py
"""

import os
import glob
import wandb
import matplotlib.pyplot as plt
from PIL import Image

from mpmath.libmp import prec_to_dps

import torch
import torch.nn as nn
import einops
from torch.optim import optimizer

import lightning.pytorch as pl
from timm.models.vision_transformer import PatchEmbed

from heavyball import ForeachSOAP

from meteolibre_model.vae.pl_model_meteofrance_dit_vae import VAEMeteoLibrePLModelDitVae
from dit_ml.dit import DiT

NORMALIZATION_FACTOR = 1.0  # 0.769


class AmbiantDiffusion(pl.LightningModule):
    """
    PyTorch Lightning module for the MeteoLibre model.

    This class wraps the SimpleConvFilmModel and integrates it with PyTorch Lightning
    for training and evaluation. It handles data loading, optimization, and defines
    the training step using a Rectified Flow approach with MSE loss.
    """

    def __init__(
        self,
        condition_size=3,
        learning_rate=1e-3,
        nb_back=4,
        nb_future=2,
        shape_image=256,
        test_dataloader=None,
        dir_save="./",
        loss_type="mse",
        parametrization="noisy",
        pretrained_vae_weight=None,
    ):
        """
        Initialize the MeteoLibrePLModel.

        Args:
            input_channels_ground (int): Number of input channels for the ground station image.
            condition_size (int): Size of the conditioning vector.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            nb_back (int, optional): Number of past frames to use as input. Defaults to 3.
            nb_future (int, optional): Number of future frames to predict. Defaults to 1.
        """
        super().__init__()

        self.nb_time_step = nb_back + nb_future

        self.encoder_model_core = DiT(
            num_patches=16 * 16 * nb_temporals,  # if 2d with flatten size
            hidden_size=384,
            depth=12,
            num_heads=8,
            use_rope=False,
            rope_dimension=3,
            max_h=16,
            max_w=16,
            max_d=nb_temporals,
        )

        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss(reduction="none")  # Rectified Flow uses MSE loss

        self.test_dataloader = test_dataloader

        self.dir_save = dir_save

        self.nb_back = nb_back
        self.nb_future = nb_future

        self.loss_type = loss_type
        self.parametrization = parametrization


    def forward(self, x_image, x_scalar):
        """
        Forward pass through the model.

        Args:
            x_image (torch.Tensor): Input image tensor.
            x_scalar (torch.Tensor): Input scalar condition tensor.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        return self.model_core(x_image, x_scalar)

    def prepare_target(self, batch):
        with torch.no_grad():
                
            radar_data = batch["radar"].unsqueeze(-1)
            groundstation_data = batch["groundstation"]

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

        return x_image, x_image_corrupt, mask_radar, mask_groundstation

    def init_prior(self, shape_target):
        return torch.randn((shape_target)).to(self.device)

    def training_step(self, batch, batch_idx):
        """
        Training step for the PyTorch Lightning module.

        This method defines the training logic for a single batch. It involves:
        1. Preparing input tensors from the batch.
        2. Projecting and embedding the ground station image.
        3. Concatenating historical images.
        4. Sampling prior noise and time variable for Rectified Flow.
        5. Interpolating between prior and future image to get the diffused sample x_t.
        6. Predicting the velocity field v_t.
        7. Calculating the target velocity field.
        8. Computing MSE loss between predicted and target velocity fields.
        9. Masking the loss for ground station data.
        10. Logging training loss.

        Args:
            batch (dict): A dictionary containing the training batch data,
                          expected to be returned by TFDataset.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Training loss for the batch.
        """
        # Assuming batch is a dictionary returned by TFDataset
        # and contains 'back_0', 'future_0', 'hour' keys

        x_image, x_image_corrupt, mask_radar, mask_groundstation = self.prepare_target(
            batch
        )
        
        target_meteo_frames = x_image[:, self.nb_back:(self.nb_back + self.nb_future)]
        input_meteo_frames = x_image[:, :self.nb_back]

        # Prior sample (simple Gaussian noise) - you can refine this prior
        prior_image = self.init_prior(target_meteo_frames.shape)

        # Time variable for Rectified Flow - sample uniformly
        t = (
            torch.rand(target_meteo_frames.shape[0], 1, 1, 1, 1)
            .type_as(target_meteo_frames)
            .to(target_meteo_frames.device)
        )  # (B, 1)

        # we create a scalar value to condition the model on time stamp
        # and hours
        x_hour = batch["hour"].clone().detach().float().unsqueeze(1)  # (B, 1)
        x_minute = batch["minute"].clone().detach().float().unsqueeze(1)  # (B, 1)

        # Simple scalar condition: hour of the day and minutes of the day. You might want to expand this.
        x_scalar = torch.cat([x_hour, x_minute, t[:, :, 0, 0, 0]], dim=1)

        # Interpolate between prior and data to get x_t
        x_t = t * target_meteo_frames + (1 - t) * prior_image

        # concat x_t with x_image_back and x_ground_station_image_previous
        input_model = torch.cat([input_meteo_frames, x_t], dim=1)

        if self.parametrization == "noisy":
            # prediction the noise
            pred = self.forward(input_model, x_scalar)

            # coefficient of ponderation for noisy parametrization
            w_t = (1 / (t + 0.0001)) ** 2

            w_t = torch.clamp(w_t, min=1.0, max=3.0)

            target = prior_image

        else:
            raise ValueError("parametrization not handled")

        # Loss is MSE between predicted and target velocity fields
        loss = self.fn_loss(pred[:, self.nb_back :, :, :, :], target)
        loss = w_t * loss  # ponderate loss

        loss = loss.mean()

        # Log the lossruff
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        optimizer = ForeachSOAP(self.parameters(), lr=self.learning_rate, foreach=False)
        return optimizer

    @torch.no_grad()
    def generate_one(self, nb_batch=1, nb_step=100, null_value=False):
        # Assuming batch is a dictionary returned by TFDataset
        # we first generate a random noise
        for batch in self.test_dataloader:
            break

        # convert to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        batch_size = batch["radar_back"].shape[0]

        # just to check create a noise value
        if null_value:
            batch["radar_back"][:, :, :, 1:] = 0.0

        target_radar_frames, input_radar_frames = self.getting_target_input_after_vae(
            batch
        )

        tmp_noise = self.init_prior(target_radar_frames.shape)

        for i in range(1, nb_step):
            # concat x_t with x_image_back and x_ground_station_image_previous
            input_model = torch.cat([input_radar_frames, tmp_noise], dim=1)

            t = i * 1.0 / nb_step

            x_scalar = torch.cat(
                [
                    batch["hour"].clone().detach().float().unsqueeze(1),
                    batch["minute"].clone().detach().float().unsqueeze(1),
                    torch.ones(batch_size, 1)
                    .type_as(batch["radar_back"])
                    .to(self.device)
                    * t,
                ],
                dim=1,
            )

            if self.parametrization == "noisy":
                noise = self.forward(input_model, x_scalar)

                velocity = (
                    1 / (t + 1e-4) * (tmp_noise - noise[:, self.nb_back :, :, :, :])
                )

                tmp_noise = tmp_noise + velocity * 1.0 / nb_step
            else:
                raise ValueError("parametrization not handled")

        return tmp_noise, target_radar_frames, input_radar_frames

    # on epoch end of training
    def on_train_epoch_end(self):
        # generate image
        self.model_core.eval()

        with torch.no_grad():
            result, target_radar_frames, input_radar_frames = self.generate_one(
                nb_batch=1, nb_step=100
            )

            full_image_result = torch.cat([input_radar_frames, result], dim=1)

            full_image_target = torch.cat(
                [input_radar_frames, target_radar_frames], dim=1
            )

            full_image_result = einops.rearrange(full_image_result, "b t c h w -> b (t h w) c")
            full_image_target = einops.rearrange(full_image_target, "b t c h w -> b (t h w) c")  

            radar_image_result = (
                self.vae.decode(full_image_result)
                .cpu()
                .numpy()
            )

            radar_image_target = (
                self.vae.decode(full_image_target)
                .cpu()
                .numpy()
            )

            # reshape first
            self.save_image(radar_image_result, name_append="result")
            self.save_image(radar_image_target, name_append="target")

            self.save_gif(radar_image_result, name_append="result_gif")
            self.save_gif(radar_image_target, name_append="target_gif")

            # now we delete all the png files (not the gif)
            for f in glob.glob(self.dir_save + "data/*.png"):
                os.remove(f)

        self.model_core.train()

    def save_gif(self, result, name_append="result", duration=10):
        nb_frame = result.shape[2]
        file_name_list = []

        for i in range(nb_frame):
            fname = (
                self.dir_save
                + f"data/{name_append}_radar_epoch_{self.current_epoch}_{i}.png"
            )

            plt.figure(figsize=(20, 20))
            plt.imshow(result[0, i, 0, :, :], vmin=-1, vmax=2)
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

    def save_image(self, result, name_append="result"):
        radar_image = result

        fname = (
            self.dir_save + f"data/{name_append}_radar_epoch_{self.current_epoch}.png"
        )

        plt.figure(figsize=(20, 20))
        plt.imshow(radar_image[0, 0, 0, :, :], vmin=-0.5, vmax=2)
        plt.colorbar()

        plt.savefig(fname, bbox_inches="tight", pad_inches=0)
        plt.close()

        # logging image into wandb
        self.logger.log_image(
            key=name_append, images=[wandb.Image(fname)], caption=[name_append]
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
    else:
        print("No valid images found to create GIF.")
