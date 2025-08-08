"""
meteolibre_model/dit/model_meteofrance_simplediffusion.py
"""

import os
import glob
import math
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from pytorch3dunet.unet3d.model import ResidualUNet3D
from meteolibre_model.dit.dit_core import DiTCore

from meteolibre_model.datasets.dataset_meteofrance_v2 import STATISTIC_SATELLITE_MEAN, STATISTIC_SATELLITE_STD

class Simple3DDiffusionModel(nn.Module):
    """
    A standard PyTorch module for the Simple3DDiffusion model.
    """

    def __init__(
        self,
        condition_size=3,
        nb_back=4,
        nb_future=2,
        nb_channels=17,
        f_maps=64,
        shape_image=256,
        parametrization="velocity",
        schedule="cosine",
    ):
        """
        Initialize the Simple3DDiffusionModel.
        """
        super().__init__()

        self.nb_time_step = nb_back + nb_future
        self.nb_channels = nb_channels
        self.f_maps = f_maps
        self.num_levels = 5

        self.model_encoder_decoder = ResidualUNet3D(
            in_channels=nb_channels,
            out_channels=nb_channels,
            final_sigmoid=True,
            f_maps=self.f_maps,
            layer_order="gcr",
            num_groups=8,
            num_levels=self.num_levels,
            is_segmentation=True,
            conv_padding=1,
            pool_kernel_size=(1, 2, 2),
        )
        
        dit_hidden_size = 768
        dit_heads = 12

        self.model_core = DiTCore(
            self.nb_time_step,
            hidden_size=dit_hidden_size,
            depth=12,
            num_heads=dit_heads,
            patch_size=2,
            out_channels=1024,
            in_channels=1024,
        )

        self.film_layer = nn.Sequential(
            nn.Linear(condition_size, 384, bias=True),
            nn.SiLU(),
            nn.Linear(
                384,
                2 * sum([self.f_maps * 2**i for i in range(self.num_levels)]),
                bias=True,
            ),
        )

        self.nb_back = nb_back
        self.nb_future = nb_future
        self.shape_image = shape_image
        self.parametrization = parametrization
        self.schedule = schedule

    def forward(self, x_image, x_scalar):
        """
        Forward pass through the model.
        """
        x = einops.rearrange(
            x_image, "batch_size nb_frame c h w -> batch_size c nb_frame h w"
        )

        film_params = self.film_layer(x_scalar)

        sum_ch = sum([self.f_maps * 2**i for i in range(self.num_levels)])
        gamma_params = film_params[:, :sum_ch]
        beta_params = film_params[:, sum_ch:]
        offset = 0
        
        encoders_features = []
        for idx, encoder in enumerate(self.model_encoder_decoder.encoders):
            x = encoder(x)

            ch = self.f_maps * (2 ** idx)
            gamma = gamma_params[:, offset:offset + ch].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            beta = beta_params[:, offset:offset + ch].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x = (1. + gamma) * x + beta
            offset += ch

            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        x = self.model_core(x, x_scalar)

        for decoder, encoder_features in zip(
            self.model_encoder_decoder.decoders, encoders_features
        ):
            x = decoder(encoder_features, x)

        x = self.model_encoder_decoder.final_conv(x)

        x = einops.rearrange(
            x, "batch_size c nb_frame h w -> batch_size nb_frame c h w"
        )

        return x[:, self.nb_back :]

    def prepare_target(self, batch, device):
        with torch.no_grad():
            radar_data = batch["radar"].unsqueeze(-1)[:, : self.nb_time_step]
            groundstation_data = batch["groundstation"][:, : self.nb_time_step]
            
            satellite_data = batch["satellite"][:, : self.nb_time_step].permute(0, 1, 3, 4, 2).float()
            
            stat_sat_mean = STATISTIC_SATELLITE_MEAN.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
            stat_sat_std = STATISTIC_SATELLITE_STD.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
            
            satellite_data = (satellite_data - stat_sat_mean) / stat_sat_std
            
            groundheight = batch["ground_height"].unsqueeze(-1).unsqueeze(1).float()
            landcover = batch["landcover"].unsqueeze(1).float()

            groundheight = groundheight.repeat(1, self.nb_time_step, 1, 1, 1)
            landcover = landcover.repeat(1, self.nb_time_step, 1, 1, 1)

            groundstation_data = torch.where(
                groundstation_data == -100, -4, groundstation_data
            )

            mask_radar = torch.ones_like(radar_data).bool()
            mask_groundstation = groundstation_data != -4

            groundstation_data_corrupt = torch.where(
                torch.rand_like(groundstation_data) > 0.3, groundstation_data, -4
            )

            x_image = torch.cat(
                (groundheight, landcover, radar_data, groundstation_data, satellite_data), dim=-1
            )
            x_image = x_image.permute(0, 1, 4, 2, 3)

            x_image_corrupt = torch.cat(
                (groundheight, landcover, radar_data, groundstation_data_corrupt),
                dim=-1,
            )
            x_image_corrupt = x_image_corrupt.permute(
                0, 1, 4, 2, 3
            )

            mask_radar = mask_radar.permute(0, 1, 4, 2, 3)
            mask_groundstation = mask_groundstation.permute(0, 1, 4, 2, 3)[
                :, self.nb_back : (self.nb_back + self.nb_future)
            ]

        return x_image, x_image_corrupt, mask_radar, mask_groundstation

    def init_prior(self, shape_target, device):
        return torch.randn((shape_target)).to(device)
    
    def get_proper_schedule(self, t):
        if self.schedule == "linear":
            return t, torch.ones_like(t)
        elif self.schedule == "cosine":
            return 1. - torch.cos(math.pi / 2 * t ** 3) ** 2, (3 * torch.pi / 2) * t**2 * torch.sin(torch.pi * t**3)
        else:
            raise ValueError("scheduler not handle")

    def compute_loss(self, batch, device):
        x_image, _, _, mask_groundstation = self.prepare_target(batch, device)
        batch_size = x_image.shape[0]

        target_meteo_frames = x_image[:, self.nb_back : (self.nb_back + self.nb_future)]
        input_meteo_frames = x_image[:, : self.nb_back]

        prior_image = self.init_prior(target_meteo_frames.shape, device)

        t = stratified_uniform_sample(batch_size, device=device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        x_hour = batch["hour"].clone().detach().float().unsqueeze(1)
        x_minute = batch["minute"].clone().detach().float().unsqueeze(1)
        x_scalar = torch.cat([x_hour, x_minute, t[:, :, 0, 0, 0]], dim=1)

        schedule, _ = self.get_proper_schedule(t)
        x_t = schedule * target_meteo_frames + (1 - schedule) * prior_image
        input_model = torch.cat([input_meteo_frames, x_t], dim=1)

        pred = self.forward(input_model, x_scalar)

        if self.parametrization == "noisy":
            target = prior_image
        elif self.parametrization == "velocity":
            target = target_meteo_frames - prior_image
        else:
            raise ValueError("parametrization not handled")

        loss_radar = F.mse_loss(pred[:, :, [5], :, :], target[:, :, [5], :, :], reduction="mean")
        loss_groundstation = F.mse_loss(
            pred[:, :, 6:13, :, :][mask_groundstation],
            target[:, :, 6:13, :, :][mask_groundstation],
            reduction="mean",
        )
        loss_satellite = F.mse_loss(pred[:, :, 13:, :, :], target[:, :, 13:, :, :], reduction="mean")
        loss_ground = F.mse_loss(pred[:, :, :5, :, :], target[:, :, :5, :, :], reduction="mean")

        total_loss = (
            loss_radar * 1.0
            + loss_groundstation * 0.3
            + loss_ground * 0.01 
            + loss_satellite * 0.5
        )
        
        losses = {
            "total_loss": total_loss,
            "loss_radar": loss_radar,
            "loss_groundstation": loss_groundstation,
            "loss_satellite": loss_satellite,
            "loss_ground": loss_ground,
        }

        return losses

def create_gif_pillow(image_paths, output_path, duration=100):
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
            loop=0,
        )

def stratified_uniform_sample(batch_size, device=None, dtype=torch.float32):
    if device is None:
        device = torch.device("cpu")
    t = torch.arange(batch_size, device=device, dtype=dtype)
    offsets = torch.rand(batch_size, device=device, dtype=dtype)
    time_stamps = (t + offsets) / batch_size
    return time_stamps
