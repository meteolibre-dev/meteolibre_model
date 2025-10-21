import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
import sys
from datetime import datetime, timedelta
import imageio
import matplotlib.pyplot as plt
import h5py
import yaml

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

# Load config
config_path = os.path.join(project_root, "meteolibre_model/config/configs.yml")
with open(config_path) as f:
    config = yaml.safe_load(f)
params = config['model_v0_mtg_lightning_shortcut']
context_frames = params['model']['context_frames']

c_sat = params['model']['sat_in_channels']
c_lightning = params['model']['kpi_in_channels']
nb_channels = c_sat + c_lightning

def create_video(forecast_dir, data_file, output_dir, forecast_steps):
    """
    Generates videos comparing forecasts to ground truth for each channel.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load h5 data
    with h5py.File(data_file, 'r') as hf:
        sat_data = hf['sat_data'][:]
        lightning_data = hf['lightning_data'][:]
        num_frames = hf.attrs['num_frames']

    # Parse initial_date from filename
    filename = os.path.basename(data_file)
    date_str = filename.split('_full.h5')[0]  # "2025-10-14_04-00"
    date_part, time_part = date_str.split('_')
    year, month, day = map(int, date_part.split('-'))
    hour, minute = map(int, time_part.split('-'))
    initial_date = datetime(year, month, day, hour, minute) - timedelta(minutes=forecast_steps*10)

    for channel in range(nb_channels):
        images = []

        vmin=-4 #min(np.min(forecast_channel_data), np.min(true_channel_data))
        vmax=None

        for step in range(forecast_steps):

            
            prediction_date = initial_date + timedelta(minutes=10 * (step + 1))

            # Load forecast data
            forecast_file = os.path.join(forecast_dir, f"forecast_{prediction_date.strftime('%Y%m%d%H%M')}.npz")
            if not os.path.exists(forecast_file):
                print(f"Warning: Forecast file not found for {prediction_date}, skipping.")
                continue

            forecast_data = np.load(forecast_file)
            sat_forecast = forecast_data['sat_forecast']
            lightning_forecast = forecast_data['lightning_forecast']
            forecast_full = np.concatenate([sat_forecast, lightning_forecast], axis=0)  # (nb_channels, H, W)
            forecast_channel_data = forecast_full[channel]

            # Load true data
            true_index = int((prediction_date - initial_date).total_seconds() / 600)
            if true_index >= num_frames:
                print(f"Warning: True index {true_index} out of range, skipping.")
                continue

            sat_true = sat_data[true_index]
            lightning_true = lightning_data[true_index]
            true_full = np.concatenate([sat_true, lightning_true], axis=0)
            true_channel_data = true_full[channel]

            if not vmax:
                vmax = max(np.max(forecast_channel_data), np.max(true_channel_data))

            # Generate a side-by-side comparison image
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(forecast_channel_data, cmap='plasma', vmin=vmin, vmax=vmax)
            axes[0].set_title(f'Forecast - Channel {channel} - {prediction_date.strftime("%Y-%m-%d %H:%M")}')
            axes[0].axis('off')
            
            axes[1].imshow(true_channel_data, cmap='plasma', vmin=vmin, vmax=vmax)
            axes[1].set_title(f'True - Channel {channel} - {prediction_date.strftime("%Y-%m-%d %H:%M")}')
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
    parser = argparse.ArgumentParser(description="Generate videos to compare forecasts with ground truth from H5.")
    parser.add_argument(
        "--forecast_dir",
        type=str,
        default="generated_forecasts_shortcut",
        help="Directory containing the forecast .npz files.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="../dataset/data_inference_full/2025-10-14_04-00_full.h5",
        help="Path to the HDF5 file containing the ground truth.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="forecast_videos_shortcut",
        help="Directory to save the generated videos.",
    )
    parser.add_argument(
        "--forecast_steps",
        type=int,
        default=18,
        help="Number of forecast steps to visualize.",
    )
    
    args = parser.parse_args()
    
    create_video(args.forecast_dir, args.data_file, args.output_dir, args.forecast_steps)

if __name__ == "__main__":
    main()