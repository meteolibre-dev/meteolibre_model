"""
Evaluation script for MeteoLibre using Rectified Flow.
This script evaluates a trained rectified flow model using the MeteoLibreMapDataset and DualUNet3DFiLM.
"""

import sys
import os
import torch
from torch.utils.data import DataLoader
import yaml
import argparse

# Add project root to sys.path
project_root = os.path.abspath("/workspace/meteolibre_model/")
sys.path.insert(0, project_root)

from meteolibre_model.dataset.dataset_mtg_lightning import MeteoLibreMapDataset
from meteolibre_model.models.unet3d_film_dual import DualUNet3DFiLM
from meteolibre_model.evaluate.eval_lightning import evaluate_model
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description="Evaluate Rectified Flow model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights file (.safetensors)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples per input for probabilistic metrics")
    parser.add_argument("--lightning_threshold", type=float, default=0.05, help="Threshold for binarizing lightning events")
    parser.add_argument("--num_steps_list", type=str, default="16,64,128", help="Comma-separated list of inference steps to evaluate")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to dataset (overrides config)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_type", type=str, default="shortcut", help="model_type (shortcut or standart)")


    args = parser.parse_args()

    # Parse num_steps_list
    num_steps_list = [int(x.strip()) for x in args.num_steps_list.split(",")]

    # Load config
    config_path = os.path.join(project_root, "meteolibre_model/config/configs.yml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if args.model_type == "shortcut":
        params = config['model_v0_mtg_lightning_shortcut']
    else:
        params = config['model_v0_mtg_lightning']

    # Override dataset path if provided
    dataset_path = args.dataset_path or params['dataset_path']

    # Initialize dataset (test split)
    dataset = MeteoLibreMapDataset(
        localrepo=dataset_path,
        cache_size=4,
        seed=args.seed, # Assuming the dataset has a test split
    )

    # Initialize DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize model
    model_params = params["model"]
    model = DualUNet3DFiLM(**model_params)

    # Load model weights
    if args.model_path.endswith('.safetensors'):
        state_dict = load_file(args.model_path)
    else:
        state_dict = torch.load(args.model_path, map_location='cpu')

    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    

    print(f"Loaded model from {args.model_path}")
    print(f"Evaluating on {len(dataset)} samples with batch size {args.batch_size}")
    print(f"Steps to evaluate: {num_steps_list}")
    print(f"Number of samples per input: {args.num_samples}")

    # Evaluate model
    metrics = evaluate_model(
        model=model,
        test_loader=dataloader,
        device=args.device,
        num_steps_list=num_steps_list,
        num_samples=args.num_samples,
        lightning_threshold=args.lightning_threshold,
        model_type=args.model_type,
    )

    # Print results
    print("\nEvaluation Results:")
    for steps, step_metrics in metrics.items():
        print(f"\nSteps: {steps}")
        for metric_name, value in step_metrics.items():
            print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()