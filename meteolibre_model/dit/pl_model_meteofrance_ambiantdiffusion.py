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


from heavyball import ForeachSOAP

from meteolibre_model.dit.ASFT import ASFTEncoder, ASFTDecoder


DEFAULT_GS_VALUE = -4.0


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
        nb_channels=13,
        shape_image=256,
        nb_pixels_selection=1024,
        test_dataloader=None,
        dir_save="./",
        loss_type="mse",
        parametrization="noisy",
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
        self.nb_pixels_selection = nb_pixels_selection
        self.nb_channels = nb_channels

        # projection to hidden size
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss(reduction="none")  # Rectified Flow uses MSE loss

        self.encoder = ASFTEncoder(in_channels=nb_channels)
        self.decoder = ASFTDecoder(embed_dim=384, num_heads=8, output_dim=nb_channels)

        self.test_dataloader = test_dataloader

        self.dir_save = dir_save

        self.nb_back = nb_back
        self.nb_future = nb_future
        self.shape_image = shape_image

        self.loss_type = loss_type
        self.parametrization = parametrization

        # projection decoding
        self.decode_projection = nn.Linear(nb_channels + 3, 384)

    def forward(self, x_image, x_scalar, input_decode):
        """
        Forward pass through the model.

        Args:
            x_image (torch.Tensor): Input image tensor.
            x_scalar (torch.Tensor): Input scalar condition tensor.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        # 1. Pass x_images thought DiT/ViT setup
        # 2. Pass input_decode though crossattention layer
        encoded_values = self.encoder(x_image, x_scalar)

        input_decode = self.decode_projection(input_decode)
        decoded_values = self.decoder(input_decode, encoded_values, x_scalar)

        return decoded_values

    def prepare_target(self, batch):
        with torch.no_grad():
            radar_data = batch["radar"].unsqueeze(-1)[:, : self.nb_time_step]
            groundstation_data = batch["groundstation"][:, : self.nb_time_step]

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
                (groundheight, landcover, radar_data, groundstation_data), dim=-1
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
        x_t = t * target_meteo_frames + (1 - t) * prior_image

        # concat x_t with x_image_back and x_ground_station_image_previous
        input_model = torch.cat([input_meteo_frames, x_t], dim=1)

        # here we should randomly select element (nb_pixels_selection) in target_meteo_frames and retrieve their position (t, x, y)
        input_decode_noisy, input_decode, target_noise = select_random_points(
            x_t, target_meteo_frames, prior_image, self.nb_pixels_selection
        )

        if self.parametrization == "noisy":
            # prediction the noise
            pred = self.forward(input_model, x_scalar, input_decode_noisy)

            # coefficient of ponderation for noisy parametrization
            w_t = (1 / (t + 0.0001)) ** 2

            w_t = torch.clamp(w_t, min=1.0, max=3.0)

        else:
            raise ValueError("parametrization not handled")

        # Loss is MSE between predicted and target velocity fields
        loss = self.criterion(pred, target_noise)
        loss = w_t * loss  # ponderate loss

        mask_all = torch.where(input_decode != -4, 1, 0)

        loss = loss * mask_all

        loss = loss.mean() / mask_all.float().mean()

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

            if self.parametrization == "noisy":
                noisy_point = select_all_points(tmp_noise)
                noise = self.forward(input_model, x_scalar, noisy_point)

                # 3. Reshape the flat list of points back into the video grid
                # (B, T*W*H, C) -> (B, T, W, H, C)
                reshaped_noise = noise.view(
                    batch_size,
                    self.nb_future,
                    self.shape_image,
                    self.shape_image,
                    self.nb_channels,
                )

                # 4. Permute the dimensions to match the standard video format (B, T, C, W, H)
                # (B, T, W, H, C) -> (B, T, C, W, H)
                reshaped_noise = reshaped_noise.permute(0, 1, 4, 2, 3)

                velocity = 1 / (t + 1e-4) * (tmp_noise - reshaped_noise)

                tmp_noise = tmp_noise + velocity * 1.0 / nb_step
            else:
                raise ValueError("parametrization not handled")

        return tmp_noise, target_meteo_frames, input_meteo_frames

    # on epoch end of training
    def on_train_epoch_end(self):
        # generate image
        self.eval()

        with torch.no_grad():
            result, target_radar_frames, input_radar_frames = self.generate_one(
                nb_batch=1, nb_step=100
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
        
        img = plt.imread(fname)[:, :, :3]
        img = img.transpose((2, 0, 1))

        # logging image into wandb
        self.logger.experiment.add_image(
                name_append, img, global_step=self.global_step
            )


def select_random_points(
    video_tensor_noisy: torch.Tensor, video_tensor_target: torch.Tensor, pur_noise: torch.Tensor, num_points: int
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
    gathered_pur_noise = pur_noise[
        b_indices, t_indices, :, w_indices, h_indices
    ]

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
