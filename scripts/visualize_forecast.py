import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
import sys
from datetime import datetime, timedelta
import imageio
import matplotlib.pyplot as plt

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

DEFAULT_VALUE = -1000

from meteolibre_model.diffusion.utils import MEAN_CHANNEL, STD_CHANNEL

def denormalize(data, device='cpu'):
    """
    Denormalize the data using precomputed mean and std.
    """
    mean = MEAN_CHANNEL.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
    std = STD_CHANNEL.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
    return data * std + mean

def create_video(forecast_dir, ground_truth_dir, output_dir, forecast_steps, nb_channels):
    """
    Generates videos comparing forecasts to ground truth for each channel.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Hardcoded initial date based on tiled_inference.py
    initial_date = datetime(2025, 8, 11, 8, 0)
    
    # Starting from the first forecast step
    start_forecast_date = initial_date + timedelta(minutes=10)


    for channel in range(nb_channels):
        images = []
        for step in range(forecast_steps):
            current_date = start_forecast_date + timedelta(minutes=10 * step)
            
            # Load forecast data
            forecast_file = os.path.join(forecast_dir, f"forecast_{current_date.strftime('%Y%m%d%H%M')}.npz")
            if not os.path.exists(forecast_file):
                print(f"Warning: Forecast file not found for {current_date}, skipping.")
                continue
            
            forecast_data = np.load(forecast_file)['forecast']
            forecast_tensor = torch.from_numpy(forecast_data).unsqueeze(0)
            
            # Denormalize the forecast data
            denormalized_forecast = denormalize(forecast_tensor)
            forecast_channel_data = denormalized_forecast[0, channel, :, :].numpy()

            # Load ground truth data
            gt_file_name_dt = initial_date + timedelta(minutes=10 * (step + 4)) # We add 4 because we start with 4 frames
            gt_file = os.path.join(ground_truth_dir, f"{gt_file_name_dt.strftime('%Y%m%d%H%M')}.npy")
            
            if not os.path.exists(gt_file):
                print(f"Warning: Ground truth file not found for {current_date}, skipping.")
                continue

            ground_truth_data = np.load(gt_file)
            ground_truth_channel_data = ground_truth_data[0, channel, :, :]

            mean_withoutmask = np.min(forecast_channel_data)
            ground_truth_channel_data = np.where(ground_truth_channel_data <= DEFAULT_VALUE, mean_withoutmask, ground_truth_channel_data)

            vmin = mean_withoutmask
            vmax = np.max(forecast_channel_data)

            # Generate a side-by-side comparison image
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(forecast_channel_data, cmap='plasma', vmin=vmin, vmax=vmax)
            axes[0].set_title(f'Forecast - Channel {channel} - {current_date}')
            axes[0].axis('off')
            
            axes[1].imshow(ground_truth_channel_data, cmap='plasma', vmin=vmin, vmax=vmax)
            axes[1].set_title(f'Ground Truth - Channel {channel} - {current_date}')
            axes[1].axis('off')

            plt.tight_layout()
            
            # Convert plot to image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            images.append(image[..., :3])
            plt.close(fig)

        if images:
            video_path = os.path.join(output_dir, f'channel_{channel}_comparison.mp4')
            imageio.mimsave(video_path, images, fps=1)
            print(f'Saved video for channel {channel} to {video_path}')

def main():
    parser = argparse.ArgumentParser(description="Generate videos to compare forecasts with ground truth.")
    parser.add_argument("--forecast_dir", type=str, default="generated_forecasts", help="Directory containing the forecast .npz files.")
    parser.add_argument("--ground_truth_dir", type=str, default="../data_inference/2025081108", help="Directory containing the ground truth .npy files.")
    parser.add_argument("--output_dir", type=str, default="forecast_videos", help="Directory to save the generated videos.")
    parser.add_argument("--forecast_steps", type=int, default=12, help="Number of forecast steps to visualize.")
    parser.add_argument("--nb_channels", type=int, default=12, help="Number of channels to visualize.")
    
    args = parser.parse_args()
    
    create_video(args.forecast_dir, args.ground_truth_dir, args.output_dir, args.forecast_steps, args.nb_channels)

if __name__ == "__main__":
    main()
