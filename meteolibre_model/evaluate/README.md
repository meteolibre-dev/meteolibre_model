# Weather Forecasting Model Horizon Evaluation

This directory contains a comprehensive evaluation system for weather forecasting models that focuses on **temporal horizon performance** using Mean Squared Error (MSE) as the primary metric.

## Overview

The evaluation system is designed to:
- Test model performance at various forecasting horizons (1-step, 2-step, 3-step, etc.)
- Use tiled inference for large-scale evaluation
- Provide clean MSE metrics for satellite and lightning channels
- Support comparison between different model configurations
- Generate detailed analysis and visualizations

## Files Structure

```
meteolibre_model/evaluate/
├── __init__.py                    # Package initialization
├── eval_lightning.py             # Existing evaluation functions (comprehensive metrics)
├── horizon_evaluation.py         # Main horizon evaluation class and functions
├── evaluation_utils.py           # Utility functions for analysis and comparison
├── demo_evaluation.py            # Demo scripts showing usage examples
└── README.md                     # This file
```

## Quick Start

### 1. Basic Usage

```python
from meteolibre_model.evaluate.horizon_evaluation import HorizonEvaluator
from meteolibre_model.evaluate.evaluation_utils import quick_evaluate

# Quick evaluation - just one line!
results = quick_evaluate(
    model_path="models/your_model.safetensors",
    data_file="path/to/your/test_data.h5",
    horizons=[1, 2, 3, 6, 12, 18],
    patch_size=128,
    denoising_steps=64,
    device='cuda'
)
```

### 2. Command Line Usage

```bash
# Basic horizon evaluation
python meteolibre_model/evaluate/horizon_evaluation.py \
    --model_path models/your_model.safetensors \
    --data_file path/to/test_data.h5 \
    --horizons 1 2 3 6 12 18 \
    --patch_size 128 \
    --denoising_steps 64 \
    --output_dir my_evaluation_results

# Compare multiple horizons with fewer steps for speed
python meteolibre_model/evaluate/horizon_evaluation.py \
    --model_path models/your_model.safetensors \
    --data_file path/to/test_data.h5 \
    --horizons 1 3 6 12 \
    --denoising_steps 32 \
    --batch_size 32
```

### 3. Run Demo

```bash
# Run the demo to see all features
python meteolibre_model/evaluate/demo_evaluation.py
```

## Key Features

### 🎯 **Horizon-Based Evaluation**
- Test performance at specific temporal distances (1, 2, 3, 6, 12, 18 steps ahead)
- Autoregressive generation for multi-step forecasting
- Track performance degradation over time

### 📊 **MSE-Centric Metrics**
- Primary metric: Mean Squared Error for both satellite and lightning channels
- Separate tracking for satellite imagery and lightning data
- Clear performance degradation analysis

### 🧩 **Tiled Inference**
- Supports large-scale evaluation with memory-efficient tiled processing
- Handles high-resolution weather data without memory constraints
- Dual-grid approach for better coverage

### 🔄 **Autoregressive Support**
- Models the real-world forecasting scenario
- Uses generated predictions as context for next steps
- Handles long-term prediction chains

### 📈 **Analysis & Visualization**
- MSE vs horizon plots
- Model comparison capabilities
- Stability analysis across horizons
- Individual prediction saving for detailed analysis

## Core Classes and Functions

### `HorizonEvaluator` Class

Main evaluation class that handles the complete evaluation pipeline.

```python
from meteolibre_model.evaluate.horizon_evaluation import HorizonEvaluator

# Initialize
evaluator = HorizonEvaluator()

# Load model
model = evaluator.load_model("path/to/model.safetensors", device="cuda")

# Run evaluation
results = evaluator.evaluate_horizons(
    model=model,
    data_file="path/to/data.h5",
    horizons=[1, 2, 3, 6, 12, 18],
    patch_size=128,
    denoising_steps=64,
    batch_size=32,
    device="cuda"
)
```

**Key Methods:**
- `load_model()` - Load and prepare the model
- `evaluate_horizons()` - Main evaluation function
- `tiled_inference_single_step()` - Single-step tiled inference
- `compute_mse()` - Calculate MSE metrics

### Utility Functions

```python
from meteolibre_model.evaluate.evaluation_utils import (
    quick_evaluate,           # One-line evaluation
    compare_models,           # Compare multiple models
    plot_mse_vs_horizon,      # Create visualizations
    analyze_stability         # Stability analysis
)

# Quick evaluation
results = quick_evaluate(model_path, data_file)

# Model comparison
results = compare_models([
    {'name': 'model_v1', 'path': 'models/v1.safetensors'},
    {'name': 'model_v2', 'path': 'models/v2.safetensors'}
], data_file)

# Stability analysis
stability = analyze_stability(results)
```

## Input Data Format

The evaluation system expects HDF5 files with the following structure:

```
data.h5
├── sat_data: (frames, channels, height, width)
├── lightning_data: (frames, 1, height, width)
└── Attributes:
    ├── num_frames: int
    ├── num_sat_channels: int
    ├── num_lightning_channels: int
    ├── target_height: int
    ├── target_width: int
    ├── transform: array
    ├── epsg: int
    └── (other metadata)
```

## Output Format

Each evaluation generates:

### 1. Summary Results (`evaluation_summary.npz`)
```python
{
    'horizons': [1, 2, 3, 6, 12, 18],
    'satellite_mse': [0.001, 0.002, 0.003, 0.005, 0.008, 0.012],
    'lightning_mse': [0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
    'average_sat_mse': 0.0052,
    'average_lightning_mse': 0.275,
    'metadata': {...}
}
```

### 2. Individual Predictions (`horizon_X_prediction.npz`)
```python
{
    'satellite_prediction': array(shape=(channels, height, width)),
    'lightning_prediction': array(shape=(1, height, width)),
    'satellite_ground_truth': array(shape=(channels, height, width)),
    'lightning_ground_truth': array(shape=(1, height, width)),
    'horizon': X
}
```

### 3. Visualizations
- `mse_vs_horizon.png` - MSE performance vs forecasting horizon
- `model_comparison.png` - Comparison plots for multiple models

### 4. Analysis Reports
- `stability_analysis.json` - Detailed stability metrics
- `model_comparison_summary.npz` - Multi-model comparison results

## Parameters and Tuning

### Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `horizons` | Steps ahead to forecast | [1, 2, 3, 6, 12, 18] |
| `patch_size` | Size of tiles for inference | 128, 256 |
| `denoising_steps` | Number of diffusion steps | 32 (fast) to 128 (quality) |
| `batch_size` | Patches processed simultaneously | 32, 64, 128 |
| `device` | Computing device | 'cuda', 'cpu' |

### Performance vs Quality Trade-offs

**For Fast Evaluation:**
```python
horizons=[1, 3, 6]
denoising_steps=32
batch_size=64
```

**For High Quality:**
```python
horizons=[1, 2, 3, 6, 12, 18]
denoising_steps=128
batch_size=32
```

## Advanced Usage

### Custom Model Integration

```python
class CustomHorizonEvaluator(HorizonEvaluator):
    def __init__(self, custom_config_path):
        super().__init__(custom_config_path)
        self.custom_params = self.load_custom_params()
    
    def load_custom_params(self):
        # Load your custom model parameters
        pass
    
    def custom_evaluation_hook(self, results):
        # Add custom analysis
        return enhanced_results
```

### Batch Evaluation Script

```python
#!/usr/bin/env python3
import os
from meteolibre_model.evaluate.evaluation_utils import quick_evaluate

# Evaluate multiple models
model_dir = "models/"
data_file = "data/test_set.h5"

for model_file in os.listdir(model_dir):
    if model_file.endswith('.safetensors'):
        model_path = os.path.join(model_dir, model_file)
        output_dir = f"eval_{model_file.replace('.safetensors', '')}"
        
        results = quick_evaluate(
            model_path=model_path,
            data_file=data_file,
            horizons=[1, 3, 6, 12],
            denoising_steps=64,
            output_dir=output_dir
        )
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` or `patch_size`
   - Use fewer horizons
   - Enable gradient checkpointing

2. **Slow Inference**
   - Reduce `denoising_steps` for faster evaluation
   - Increase `batch_size` (if memory allows)
   - Use fewer horizons

3. **Data Loading Errors**
   - Check HDF5 file format matches expected structure
   - Verify file paths are correct
   - Ensure sufficient disk space

4. **Model Loading Failures**
   - Verify model file exists and is accessible
   - Check model architecture matches evaluation code
   - Ensure model was trained with compatible settings

### Performance Monitoring

```python
import time
from meteolibre_model.evaluate.evaluation_utils import quick_evaluate

start_time = time.time()
results = quick_evaluate(model_path, data_file, horizons=[1, 2, 3])
end_time = time.time()

print(f"Evaluation took {end_time - start_time:.2f} seconds")
print(f"Average time per horizon: {(end_time - start_time) / 3:.2f} seconds")
```

## Integration with Existing Code

The new evaluation system is designed to work alongside your existing evaluation infrastructure:

```python
# Import both evaluation systems
from meteolibre_model.evaluate.eval_lightning import evaluate_model  # Existing comprehensive metrics
from meteolibre_model.evaluate.horizon_evaluation import HorizonEvaluator  # New horizon-specific evaluation

# Run both evaluations
comprehensive_results = evaluate_model(model, test_loader, device)  # All metrics
horizon_results = HorizonEvaluator().evaluate_horizons(model, data_file, horizons)  # MSE-focused

# Combine results
combined_analysis = {
    'comprehensive': comprehensive_results,
    'horizon_mse': horizon_results
}
```

## Best Practices

1. **Start with Quick Evaluation**: Use fewer horizons and steps for initial testing
2. **Monitor Memory Usage**: Large patch sizes and batch sizes can exhaust GPU memory
3. **Save Individual Predictions**: Useful for debugging and detailed analysis
4. **Compare Models**: Use the comparison utilities to track improvements
5. **Track Stability**: Monitor performance degradation across horizons
6. **Use Version Control**: Save evaluation results with model versions

## Contributing

To extend the evaluation system:

1. **Add New Metrics**: Extend the `compute_metrics()` function
2. **New Visualizations**: Add plotting functions to `evaluation_utils.py`
3. **Custom Evaluators**: Inherit from `HorizonEvaluator` for specialized needs
4. **Integration**: Update `demo_evaluation.py` with usage examples

## License

This evaluation system is part of the meteolibre_model project. See the main project license for details.