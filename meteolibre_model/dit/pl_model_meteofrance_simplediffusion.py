"""
meteolibre_model/meteolibre_model/pl_model.py
"""

import os
import glob
import wandb
import math
import matplotlib.pyplot as plt
from PIL import Image

from mpmath.libmp import prec_to_dps

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch.optim import optimizer

import lightning.pytorch as pl

from heavyball import ForeachSOAP

# from diffusers import AutoencoderKLLTXVideo
from pytorch3dunet.unet3d.model import ResidualUNet3D
from meteolibre_model.dit.dit_core import DiTCore

from meteolibre_model.datasets.dataset_meteofrance_v2 import STATISTIC_SATELLITE_MEAN, STATISTIC_SATELLITE_STD

DEFAULT_GS_VALUE = -4.0


class Simple3DDiffusion(pl.LightningModule):
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
        nb_channels=17,
        f_maps=64,
        shape_image=256,
        test_dataloader=None,
        dir_save="./",
        loss_type="mse",
        parametrization="velocity",
        schedule="cosine",
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
        self.nb_channels = nb_channels

        # projection to hidden size
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss(reduction="none")  # Rectified Flow uses MSE loss

        self.f_maps = f_maps
        self.num_levels = 5

        self.model_encoder_decoder = ResidualUNet3D(
            nb_channels,
            nb_channels,
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

        # film layer projection
        self.film_layer = nn.Sequential(
            nn.Linear(condition_size, 384, bias=True),
            nn.SiLU(),
            nn.Linear(
                384,
                2 * sum([self.f_maps * 2**i for i in range(self.num_levels)]),
                bias=True,
            ),
        )

        self.test_dataloader = test_dataloader

        self.dir_save = dir_save

        self.nb_back = nb_back
        self.nb_future = nb_future
        self.shape_image = shape_image

        self.loss_type = loss_type
        self.parametrization = parametrization
        self.schedule = schedule

    def forward(self, x_image, x_scalar):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input image tensor.
            x_scalar (torch.Tensor): Input scalar condition tensor.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        x = einops.rearrange(
            x_image, "batch_size nb_frame c h w -> batch_size c nb_frame h w"
        )

        film_params = self.film_layer(x_scalar)


        sum_ch = sum([self.f_maps * 2**i for i in range(self.num_levels)])
        film_params = self.film_layer(x_scalar)
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

            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        x = self.model_core(x, x_scalar)

        # decoder part
        for decoder, encoder_features in zip(
            self.model_encoder_decoder.decoders, encoders_features
        ):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.model_encoder_decoder.final_conv(x)

        x = einops.rearrange(
            x, "batch_size c nb_frame h w -> batch_size nb_frame c h w"
        )

        return x[:, self.nb_back :]

    def prepare_target(self, batch):
        with torch.no_grad():
            radar_data = batch["radar"].unsqueeze(-1)[:, : self.nb_time_step]
            groundstation_data = batch["groundstation"][:, : self.nb_time_step]
            
            satellite_data = batch["satellite"][:, : self.nb_time_step].permute(0, 1, 3, 4, 2).float()
            
            # global statistic normalization for satellite
            stat_sat_mean = STATISTIC_SATELLITE_MEAN.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)
            stat_sat_std = STATISTIC_SATELLITE_STD.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)
            
            satellite_data = (satellite_data - stat_sat_mean) / stat_sat_std
            

            groundheight = batch["ground_height"].unsqueeze(-1).unsqueeze(1).float()
            landcover = batch["landcover"].unsqueeze(1).float()

            groundheight = groundheight.repeat(1, self.nb_time_step, 1, 1, 1)
            landcover = landcover.repeat(1, self.nb_time_step, 1, 1, 1)

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
            x_image = torch.cat(
                (groundheight, landcover, radar_data, groundstation_data, satellite_data), dim=-1
            )
            x_image = x_image.permute(0, 1, 4, 2, 3)  # (N, nb_frame, C, H, W)

            x_image_corrupt = torch.cat(
                (groundheight, landcover, radar_data, groundstation_data_corrupt),
                dim=-1,
            )
            x_image_corrupt = x_image_corrupt.permute(
                0, 1, 4, 2, 3
            )  # (N, nb_frame, C, H, W)

            mask_radar = mask_radar.permute(0, 1, 4, 2, 3)
            mask_groundstation = mask_groundstation.permute(0, 1, 4, 2, 3)[
                :, self.nb_back : (self.nb_back + self.nb_future)
            ]

        return x_image, x_image_corrupt, mask_radar, mask_groundstation

    def init_prior(self, shape_target):
        return torch.randn((shape_target)).to(self.device)
    
    def get_proper_schedule(self, t):
        if self.schedule == "linear":
            return t, torch.ones_like(t)
        elif self.schedule == "cosine":
            return 1. - torch.cos(math.pi / 2 * t ** 3) ** 2, (3 * torch.pi / 2) * t**2 * torch.sin(torch.pi * t**3)
        else:
            raise ValueError("scheduler not handle")

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

        batch_size = x_image.shape[0]

        target_meteo_frames = x_image[:, self.nb_back : (self.nb_back + self.nb_future)]
        input_meteo_frames = x_image[:, : self.nb_back]

        # Prior sample (simple Gaussian noise) - you can refine this prior
        prior_image = self.init_prior(target_meteo_frames.shape)

        # Time variable for Rectified Flow - sample uniformly
        t = (
            torch.rand(batch_size, 1, 1, 1, 1)
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
        schedule, schedule_deriv = self.get_proper_schedule(t)
        
        x_t = schedule * target_meteo_frames + (1 - schedule) * prior_image

        # concat x_t with x_image_back and x_ground_station_image_previous
        input_model = torch.cat([input_meteo_frames, x_t], dim=1)

        # prediction the noise
        pred = self.forward(input_model, x_scalar)

        if self.parametrization == "noisy":
            
            target = prior_image
            
            # coefficient of ponderation for noisy parametrization
            w_t = (1 / (t + 0.0001)) ** 2

            w_t = torch.clamp(w_t, min=1.0, max=3.0)


        elif self.parametrization == "velocity":
            
            target = target_meteo_frames - prior_image

        else:
            raise ValueError("parametrization not handled")

        reconstruction_loss_radar = F.mse_loss(
            pred[:, :, [5], :, :], target[:, :, [5], :, :], reduction="mean"
        )
        reconstruction_loss_groundstation = F.mse_loss(
            pred[:, :, 6:13, :, :][mask_groundstation],
            target[:, :, 6:13, :, :][mask_groundstation],
            reduction="mean",
        )
        
        reconstruction_loss_satellite = F.mse_loss(
            pred[:, :, 13:, :, :],
            target[:, :, 13:, :, :],
            reduction="mean",
        )
        
        reconstruction_ground = F.mse_loss(
            pred[:, :, :5, :, :], target[:, :, :5, :, :], reduction="mean"
        )

        # Loss is MSE between predicted and target velocity fields
        # loss = self.criterion(pred, prior_image)
        # loss = w_t * loss  # ponderate loss
        reconstruction_loss = (
            reconstruction_loss_radar * 3.
            + reconstruction_loss_groundstation * 0.3
            + reconstruction_ground * 0.01 
            + reconstruction_loss_satellite * 0.5
        )

        # logging data
        self.log("reconstruction_loss", reconstruction_loss)
        self.log("reconstruction_loss_radar", reconstruction_loss_radar)
        self.log("reconstruction_loss_groundstation", reconstruction_loss_groundstation)
        self.log("reconstruction_loss_ground_info", reconstruction_ground)
        self.log("reconstruction_loss_satellite", reconstruction_loss_satellite)

        return reconstruction_loss


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

        batch_size = batch["radar"].shape[0]

        x_image, x_image_corrupt, mask_radar, mask_groundstation = self.prepare_target(
            batch
        )

        batch_size = x_image.shape[0]

        target_meteo_frames = x_image[:, self.nb_back : (self.nb_back + self.nb_future)]
        input_meteo_frames = x_image[:, : self.nb_back]

        tmp_noise = self.init_prior(target_meteo_frames.shape)

        for i in range(1, nb_step):
            # concat x_t with x_image_back and x_ground_station_image_previous
            input_model = torch.cat([input_meteo_frames, tmp_noise], dim=1)

            t = i * 1.0 / nb_step

            x_scalar = torch.cat(
                [
                    batch["hour"].clone().detach().float().unsqueeze(1),
                    batch["minute"].clone().detach().float().unsqueeze(1),
                    torch.ones(batch_size, 1).type_as(batch["radar"]).to(self.device)
                    * t,
                ],
                dim=1,
            )
            
            pred = self.forward(input_model, x_scalar)

            if self.parametrization == "noisy":
                
                # error accorinf to grok
                pred_noise = pred
                velocity = (tmp_noise - pred_noise) / (1 - t + 1e-4)
                
                #velocity = 1 / (t + 1e-4) * (tmp_noise - pred) 

            elif self.parametrization == "velocity":
                
                velocity = pred
                
            else:
                raise ValueError("parametrization not handled")

            schedule, schedule_deriv = self.get_proper_schedule(torch.tensor(t))
            tmp_noise = tmp_noise + schedule_deriv.item() * velocity * 1.0 / nb_step

        return tmp_noise, target_meteo_frames, input_meteo_frames

    # on epoch end of training
    def on_train_epoch_end(self):
        # generate image
        self.eval()

        with torch.no_grad():
            result, target_radar_frames, input_radar_frames = self.generate_one(
                nb_batch=1, nb_step=200
            )

            full_image_result = torch.cat([input_radar_frames, result], dim=1)

            full_image_target = torch.cat(
                [input_radar_frames, target_radar_frames], dim=1
            )

            radar_image_result = full_image_result.cpu().numpy()
            radar_image_target = full_image_target.cpu().numpy()

            # reshape first
            self.save_image(radar_image_result, name_append="result")
            self.save_image(radar_image_target, name_append="target")

            self.save_gif(radar_image_result, name_append="result_gif")
            self.save_gif(radar_image_target, name_append="target_gif")

            # now we delete all the png files (not the gif)
            for f in glob.glob(self.dir_save + "data/*.png"):
                os.remove(f)

        self.train()

    def save_gif(self, result, name_append="result", duration=10):
        nb_frame = result.shape[1]
        file_name_list = []

        for i in range(nb_frame):
            fname = (
                self.dir_save
                + f"data/{name_append}_radar_epoch_{self.current_epoch}_{i}.png"
            )

            plt.figure(figsize=(20, 20))
            plt.imshow(result[0, i, -1, :, :], vmin=-1, vmax=2)
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
        plt.imshow(radar_image[0, 0, -1, :, :], vmin=-0.5, vmax=2)
        plt.colorbar()

        plt.savefig(fname, bbox_inches="tight", pad_inches=0)
        plt.close()

        img = plt.imread(fname)[:, :, :3]
        img = img.transpose((2, 0, 1))

        # logging image into wandb
        self.logger.experiment.add_image(name_append, img, global_step=self.global_step)


def select_random_points(
    video_tensor_noisy: torch.Tensor,
    video_tensor_target: torch.Tensor,
    pur_noise: torch.Tensor,
    num_points: int,
) -> torch.Tensor:
    """
    Selects random spatio-temporal points from a video tensor and returns their
    values and normalized coordinates.

    This function is inspired by the "Coordinate In and Value Out" concept, extended
    to a video setup. It takes a batch of videos and for each video, it randomly
    samples a set of points in the (time, width, height) space.

    Args:
        video_tensor (torch.Tensor): The input video tensor with a shape of
                                     (B, T, C, W, H), where:
                                     - B: Batch size
                                     - T: Number of frames (time dimension)
                                     - C: Number of channels (e.g., 3 for RGB)
                                     - W: Width of the frames
                                     - H: Height of the frames
        num_points (int): The number of points to randomly select from each
                          video in the batch.

    Returns:
        torch.Tensor: An output tensor of shape (B, num_points, C + 3). For each
                      selected point, the last dimension contains its C channel
                      values followed by its 3 normalized spatio-temporal
                      coordinates (t, w, h), each in the range [0, 1].
    """
    # 1. Get the shape of the input tensor and its device
    B, T, C, W, H = video_tensor_noisy.shape
    device = video_tensor_noisy.device

    # 2. Generate random indices for the spatio-temporal dimensions (T, W, H)
    # For each item in the batch, we generate `num_points` random indices.
    t_indices = torch.randint(0, T, size=(B, num_points), device=device)
    w_indices = torch.randint(0, W, size=(B, num_points), device=device)
    h_indices = torch.randint(0, H, size=(B, num_points), device=device)

    # 3. Create a batch index to ensure we gather from the correct video
    # This creates indices like [[0,0,...], [1,1,...], ...] to align with the
    # other indices for advanced indexing.
    b_indices = torch.arange(B, device=device).view(-1, 1).expand(-1, num_points)

    # 4. Gather the channel values using the generated indices (the "Value Out")
    # We use advanced indexing to pick the values at the specific (b, t, w, h)
    # locations. The ':' selects all channels for those locations.
    gathered_values = video_tensor_noisy[b_indices, t_indices, :, w_indices, h_indices]
    gathered_values_target = video_tensor_target[
        b_indices, t_indices, :, w_indices, h_indices
    ]
    gathered_pur_noise = pur_noise[b_indices, t_indices, :, w_indices, h_indices]

    # 5. Stack the indices to create coordinate vectors (the "Coordinate In")
    coords = torch.stack([t_indices, w_indices, h_indices], dim=2).float()

    # 6. Normalize the coordinates to be in the range [0, 1]
    # We divide by (Dimension - 1) to map the 0-based index range correctly.
    # We use max(1, D-1) to avoid division by zero if a dimension has size 1.
    normalizer = torch.tensor(
        [max(1, T - 1), max(1, W - 1), max(1, H - 1)], device=device, dtype=torch.float
    )

    normalized_coords = coords / normalizer

    # 7. Concatenate the gathered values and their normalized coordinates
    # This creates the final (B, num_points, C + 3) tensor.
    output_tensor_noisy = torch.cat([gathered_values, normalized_coords], dim=2)

    return output_tensor_noisy, gathered_values_target, gathered_pur_noise


def select_all_points(video_tensor: torch.Tensor) -> torch.Tensor:
    """
    Selects all spatio-temporal points from a video tensor and returns their
    values and normalized coordinates.

    This function is useful for inference or evaluation when the entire set of
    points is required, rather than a random subset.

    Args:
        video_tensor (torch.Tensor): The input video tensor with a shape of
                                     (B, T, C, W, H).

    Returns:
        torch.Tensor: An output tensor of shape (B, T*W*H, C + 3). For each
                      point, the last dimension contains its C channel values
                      followed by its 3 normalized spatio-temporal coordinates.
    """
    # 1. Get the shape of the input tensor and its device
    B, T, C, W, H = video_tensor.shape
    device = video_tensor.device

    # 2. Get all channel values by reshaping the tensor
    # (B, T, C, W, H) -> (B, C, T, W, H) -> (B, C, T*W*H) -> (B, T*W*H, C)
    all_values = video_tensor.permute(0, 2, 1, 3, 4).flatten(2).permute(0, 2, 1)

    # 3. Create a grid of all spatio-temporal coordinates
    t_coords = torch.arange(T, device=device)
    w_coords = torch.arange(W, device=device)
    h_coords = torch.arange(H, device=device)

    # Create a meshgrid of coordinates
    grid = torch.stack(
        torch.meshgrid(t_coords, w_coords, h_coords, indexing="ij"), dim=-1
    )

    # Flatten the grid to get a list of all coordinate points
    all_coords = grid.view(-1, 3).float()  # Shape: (T*W*H, 3)

    # 4. Normalize the coordinates to be in the range [0, 1]
    normalizer = torch.tensor(
        [max(1, T - 1), max(1, W - 1), max(1, H - 1)], device=device, dtype=torch.float
    )

    normalized_coords = all_coords / normalizer

    # 5. Expand the coordinates to match the batch size
    # (T*W*H, 3) -> (1, T*W*H, 3) -> (B, T*W*H, 3)
    normalized_coords = normalized_coords.unsqueeze(0).expand(B, -1, -1)

    # 6. Concatenate the values and their normalized coordinates
    output_tensor = torch.cat([all_values, normalized_coords], dim=2)

    return output_tensor


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
