
![channel_3_comparison](https://github.com/user-attachments/assets/bf4fca7d-8218-4be8-9efa-4d0b092496e8)

1km spatial resolution / 10min temporal resolution on whole europe.
Trained using https://huggingface.co/datasets/meteolibre-dev/mtg_europe_128_1km

# MeteoLibre Model

## Overview

MeteoLibre Model is a Python-based machine learning project for weather prediction. It leverages a variety of powerful libraries to process and analyze meteorological data, enabling accurate and reliable forecasting. This project is designed to be modular and extensible, allowing for easy integration of new models and data sources.

## Features

- **Data Processing:** Utilizes pandas and numpy for efficient data manipulation and preprocessing.
- **Machine Learning:** Built on PyTorch, a leading deep learning framework, for model training and inference.
- **Hugging Face Integration:** Seamlessly integrates with the Hugging Face ecosystem for access to datasets and pre-trained models.
- **Data Visualization:** Includes tools for creating insightful visualizations with matplotlib.
- **Geospatial Analysis:** Leverages pyproj for handling geospatial data and coordinate transformations.

## Model Architecture: 3D U-Net

The core model is a **3D U-Net**, implemented in `meteolibre_model/models/unet3d.py`. This architecture is particularly well-suited for tasks involving volumetric or sequential data, where spatial and temporal relationships are important.

- **Architecture:** The model follows a classic U-Net structure with an encoder, a bottleneck, and a decoder.
- **Downsampling and Upsampling:** Instead of traditional pooling layers, the model uses `DownsampleBlock3D` and `UpsampleBlock3D` blocks. These blocks perform downsampling and upsampling only on the spatial dimensions (height and width), while keeping the depth dimension intact. This is a key feature for handling meteorological data, where the depth could represent different pressure levels or time steps.
- **Residual Connections:** The model incorporates `ResNetBlock3D` blocks, which use residual connections to improve gradient flow and allow for deeper networks.
- **Input and Output:** The model takes a 5D tensor as input, with the shape `(N, C, D, H, W)`, where:
    - `N` is the batch size.
    - `C` is the number of input channels (e.g., different weather variables).
    - `D` is the depth (e.g., time steps or altitude levels).
    - `H` is the height of the spatial grid.
    - `W` is the width of the spatial grid.
The output has the same shape, with the number of channels adjusted to the desired number of output variables.

## Data Handling

The data loading is handled by the `MeteoLibreMapDataset` class in `meteolibre_model/dataset/dataset.py`. This class is designed to efficiently load and preprocess data from a collection of Parquet files.

- **Data Source:** The dataset reads data from a directory of Parquet files located in `{localrepo}/data/`.
- **Data Loading:** It uses a map-style dataset approach, where each item in the dataset corresponds to a single record in one of the Parquet files.
- **Shuffling:** The dataset implements a sophisticated shuffling mechanism that shuffles the order of Parquet files for each DataLoader worker. This ensures that each worker is likely to be processing a different file at any given time, which can improve data loading performance and randomness.
- **Caching:** An in-memory LRU (Least Recently Used) cache is used to store recently accessed DataFrames, reducing the need to read the same files from disk repeatedly.
- **Preprocessing:** The `_preprocess` method is responsible for taking a raw record from the Parquet file and preparing it for the model. This includes:
    - Reshaping the raw data into the correct 5D tensor format.
    - Calculating the sun's position (azimuth and altitude) based on the date, longitude, and latitude of the data patch. This information is then added to the input tensor as additional features.

## Installation

To get started with MeteoLibre Model, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/meteolibre-model.git
cd meteolibre-model
pip install -e .
```

## Dependencies

The project relies on the following libraries:

- **Core Libraries:**
  - pandas
  - numpy
  - torch
  - accelerate
  - tqdm
  - torchvision
- **Hugging Face:**
  - datasets
  - huggingface-hub
- **Scientific Computing:**
  - suncalc
  - scipy
  - tensorboard
  - torch-dct
  - einops
- **Visualization:**
  - matplotlib
  - imageio
- **Geospatial:**
  - pyproj

For a complete list of dependencies, please see the `pyproject.toml` file.

## Usage

To train a new model, you can use the provided training scripts. For example:

```bash
python scripts/train.py --config configs/your_config.yaml
```

For more detailed instructions on how to use the project, please refer to the documentation.

## Performance Summary

| Model          | Optimizer | Steps  | sat_mse | sat_psnr | sat_ssim | light_mae | light_precision | light_recall | light_f1 | light_iou |
|----------------|-----------|--------|---------|----------|----------|-----------|-----------------|--------------|----------|-----------|
| RF (Run 1)     | -         | 128    | 0.0952  | 28.5327  | 0.8042   | 0.0221    | 0.5482          | 0.6535       | 0.5950   | -         |
| RF (Run 2)     | -         | 128    | 0.1076  | 27.8870  | 0.8000   | 0.0221    | 0.5157          | 0.6454       | 0.5724   | -         |
| Baseline       | Persistence| baseline| 0.2368 | 24.5138  | 0.7266   | 0.0154    | 0.6714          | 0.6665       | 0.6678   | 0.1023    |
| Shortcut       | Adam      | 16     | 0.0981  | 28.3788  | 0.8106   | 0.0216    | 0.6339          | 0.5192       | 0.5686   | 0.0791    |
| Shortcut       | Adam      | 64     | 0.0983  | 28.3702  | 0.8114   | 0.0207    | 0.6609          | 0.5304       | 0.5860   | 0.0791    |
| Shortcut       | Adam      | 128    | 0.0983  | 28.3581  | 0.8112   | 0.0208    | 0.6518          | 0.5208       | 0.5769   | 0.0791    |
| Shortcut       | SOAP      | 16     | 0.0601  | 30.5008  | 0.8663   | 0.0156    | 0.8654          | 0.6958       | 0.7710   | 0.0818    |
| Shortcut       | SOAP      | 64     | 0.0606  | 30.4786  | 0.8661   | 0.0151    | 0.8658          | 0.6879       | 0.7663   | 0.0818    |
| Shortcut       | SOAP      | 128    | 0.0605  | 30.4848  | 0.8660   | 0.0151    | 0.8635          | 0.6886       | 0.7656   | 0.0818    |

Metrics from evaluation on 64x20 elements (satellite and lightning data).

## Contributing

We welcome contributions from the community! If you would like to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with a descriptive message.
4. Push your changes to your fork.
5. Submit a pull request to the main repository.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

#### Utils

Download dataset example:

```bash
hf download meteolibre-dev/weather_mtg_europe_lightning_128_1km --repo-type dataset --local-dir .
```

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```


GPU monitoring:
```bash
uvx nvitop
```