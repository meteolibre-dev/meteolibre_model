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
from torch.special import expm1

from pytorch3dunet.unet3d.model import ResidualUNet3D
from meteolibre_model.dit.dit_core import DiTCore

from meteolibre_model.datasets.dataset_meteofrance_v2 import STATISTIC_SATELLITE_MEAN, STATISTIC_SATELLITE_STD


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


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
        noise_d=64,
        image_d=256,
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
            final_sigmoid=False,
            f_maps=self.f_maps,
            layer_order="gcr",
            num_groups=8,
            num_levels=self.num_levels,
            is_segmentation=False,
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
        self.noise_d = noise_d
        self.image_d = image_d

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
    
    def logsnr_schedule_cosine(self, t, logsnr_min=-15, logsnr_max=15):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(self, t):
        logsnr_t = self.logsnr_schedule_cosine(t)
        return logsnr_t + 2 * math.log(self.noise_d / self.image_d)

    def get_logsnr(self, t):
        if self.schedule == "cosine":
            return self.logsnr_schedule_cosine(t)
        elif self.schedule == "shifted_cosine":
            return self.logsnr_schedule_cosine_shifted(t)
        else:
            raise ValueError("scheduler not handle")

    def compute_loss(self, batch, device):
        x_image, _, _, mask_groundstation = self.prepare_target(batch, device)
        batch_size = x_image.shape[0]

        x_image = self.clip(x_image) # ensure that images are stable between -3 and 3

        target_meteo_frames = x_image[:, self.nb_back : (self.nb_back + self.nb_future)]
        input_meteo_frames = x_image[:, : self.nb_back]

        eps_t = self.init_prior(target_meteo_frames.shape, device)

        t = stratified_uniform_sample(batch_size, device=device)

        logsnr_t = self.get_logsnr(t)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))

        alpha_t = alpha_t.view(-1, 1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1, 1)

        x_t = alpha_t * target_meteo_frames + sigma_t * eps_t
        input_model = torch.cat([input_meteo_frames, x_t], dim=1)

        x_hour = batch["hour"].clone().detach().float().unsqueeze(1)
        x_minute = batch["minute"].clone().detach().float().unsqueeze(1)
        logsnr_t_unsq = logsnr_t.unsqueeze(1)
        x_scalar = torch.cat([x_hour, x_minute, logsnr_t_unsq], dim=1)

        pred = self.forward(input_model, x_scalar)

        if self.parametrization == "velocity":
            eps_pred = sigma_t * x_t + alpha_t * pred
            target = eps_t
        elif self.parametrization == "noisy":
            eps_pred = pred
            target = eps_t
        else:
            raise ValueError("parametrization not handled")

        snr = torch.exp(logsnr_t).clamp(max=5)
        if self.parametrization == "velocity":
            weight = 1 / (1 + snr)
        else:
            weight = 1 / snr

        weight = weight.view(-1, 1, 1, 1, 1)
        loss_tensor = weight * (eps_pred - target) ** 2

        loss_radar = loss_tensor[:, :, [5], :, :].mean()

        if mask_groundstation.sum() != 0:
            loss_groundstation = loss_tensor[:, :, 6:13, :, :][mask_groundstation].mean()
        else:
            loss_groundstation = torch.tensor(0, device=loss_tensor.device)
            
        loss_satellite = loss_tensor[:, :, 13:, :, :].mean()
        loss_ground = loss_tensor[:, :, :5, :, :].mean()

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

    def clip(self, x):
        """
        Function to clip the input tensor x to the range [-1, 1].
        """
        return torch.clamp(x, -3, 3)

    @torch.no_grad()
    def ddpm_sampler_step(self, z_t, pred, logsnr_t, logsnr_s):
        """
        Function to perform a single step of the DDPM sampler.
        """
        c = -expm1(logsnr_t - logsnr_s)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s))

        if self.parametrization == 'velocity':
            x_pred = alpha_t * z_t - sigma_t * pred
        elif self.parametrization == 'noisy':
            x_pred = (z_t - sigma_t * pred) / alpha_t
        else:
            raise ValueError("parametrization not handled")

        x_pred = self.clip(x_pred)

        mu = alpha_s * (z_t * (1 - c) / alpha_t + c * x_pred)
        variance = (sigma_s ** 2) * c

        return mu, variance

    @torch.no_grad()
    def sample(self, input_meteo_frames, x_hour, x_minute, sampling_steps=256):
        """
        Standard DDPM sampling procedure.
        """
        device = input_meteo_frames.device
        batch_size = input_meteo_frames.shape[0]

        target_shape = (batch_size, self.nb_future, self.nb_channels, self.shape_image, self.shape_image)

        z_t = torch.randn(target_shape, device=device)

        steps = torch.linspace(1.0, 0.0, sampling_steps + 1, device=device)

        for i in range(sampling_steps):
            u_t_val = steps[i]
            u_s_val = steps[i+1]
            
            u_t = torch.full((batch_size,), u_t_val, device=device)
            
            logsnr_t = self.get_logsnr(u_t)
            logsnr_s = self.get_logsnr(torch.full((batch_size,), u_s_val, device=device))

            logsnr_t_unsq = logsnr_t.unsqueeze(1)
            x_scalar = torch.cat([x_hour, x_minute, logsnr_t_unsq], dim=1)
            
            input_model = torch.cat([input_meteo_frames, z_t], dim=1)
            pred = self.forward(input_model, x_scalar)

            logsnr_t = logsnr_t.view(-1, 1, 1, 1, 1)
            logsnr_s = logsnr_s.view(-1, 1, 1, 1, 1)

            mu, variance = self.ddpm_sampler_step(z_t, pred, logsnr_t, logsnr_s)
            
            if u_s_val > 0:
                noise = torch.randn_like(mu)
            else:
                noise = 0.
                
            z_t = mu + noise * torch.sqrt(variance)

        x_pred = self.clip(z_t)

        return x_pred

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
