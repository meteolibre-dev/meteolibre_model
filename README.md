---
language: en
tags:
- weather-forecasting
- diffusion-models
- rectified-flow
- meteorology
- pytorch
- deep-learning
license: mit
datasets:
- meteolibre
---

# MeteoLibre Rectified Flow Model

This is a rectified flow diffusion model trained for meteorological data forecasting using the MeteoLibre dataset. The model uses a 3D U-Net architecture with FiLM conditioning for efficient weather pattern generation.

## Model Description

- **Model type**: Rectified Flow Diffusion Model
- **Architecture**: 3D DC-AE U-Net with FiLM conditioning
- **Input**: Meteorological data patches (12 channels, 3D spatio-temporal)
- **Output**: Generated weather forecast data
- **Training data**: MeteoLibre meteorological dataset
- **Language(s)**: Python
- **License**: MIT

## Intended Use

This model is designed for:
- Weather pattern generation and forecasting
- Meteorological data augmentation
- Research in atmospheric science and weather prediction
- Educational purposes in machine learning for climate modeling

## Model Architecture

The model consists of:
- **UNet_DCAE_3D**: 3D convolutional U-Net with encoder-decoder architecture
- **FiLM Conditioning**: Feature-wise linear modulation for temporal context
- **Rectified Flow**: Efficient generative modeling approach
- **Input channels**: 12 (meteorological variables)
- **Output channels**: 12 (forecast variables)
- **Features**: [64, 128, 256] channel progression
- **Context frames**: 4 (temporal conditioning)

## Training

The model was trained using:
- **Framework**: PyTorch with Hugging Face Accelerate
- **Optimizer**: Adam (lr=5e-4)
- **Batch size**: 64
- **Epochs**: 200
- **Precision**: Mixed precision (bf16)
- **Distributed training**: Multi-GPU support

## Usage

### Loading the Model

```python
from safetensors.torch import load_file
import torch
from meteolibre_model.models.dc_3dunet_film import UNet_DCAE_3D

# Load model weights
state_dict = load_file("epoch_141_rectified_flow.safetensors")

# Create model
model = UNet_DCAE_3D(
    in_channels=12,
    out_channels=12,
    features=[64, 128, 256],
    context_dim=4,
    context_frames=4,
    num_additional_resnet_blocks=2
)

model.load_state_dict(state_dict)
model.eval()
```

### Inference

```python
# Example inference code
with torch.no_grad():
    generated_data = model(input_batch)
```

## Performance

The model checkpoints are saved at regular intervals:
- epoch_1_rectified_flow.safetensors through epoch_141_rectified_flow.safetensors
- Best performing checkpoints available for different training stages

## Limitations

- Model trained on specific meteorological dataset
- May not generalize to all weather patterns or regions
- Requires significant computational resources for inference
- Temporal context limited to 4 frames

## Ethical Considerations

- Weather forecasting models should be used responsibly
- Consider environmental impact of computational requirements
- Validate predictions against ground truth data
- Not intended for critical decision-making without human oversight

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{meteolibre-rectified-flow,
  title={MeteoLibre Rectified Flow Weather Forecasting Model},
  author={MeteoLibre Development Team},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/meteolibre-dev/meteolibre-rectified-flow}
}
```

## Contact

For questions or issues, please open an issue on the [MeteoLibre GitHub repository](https://github.com/meteolibre-dev/meteolibre_model).
