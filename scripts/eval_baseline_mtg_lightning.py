"""
Evaluation script for MeteoLibre baseline using previous time step.
This script evaluates a baseline model using the MeteoLibreMapDataset.
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
from meteolibre_model.evaluate.eval_lightning import evaluate_baseline


def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=40, help="Number of samples per input for probabilistic metrics")
    parser.add_argument("--lightning_threshold", type=float, default=0.05, help="Threshold for binarizing lightning events")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to dataset (overrides config)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load config
    config_path = os.path.join(project_root, "meteolibre_model/config/configs.yml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    params = config['model_v0_mtg_world_lightning_shortcut']

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

    print(f"Evaluating baseline on {len(dataset)} samples with batch size {args.batch_size}")
    print(f"Number of samples per input: {args.num_samples}")

    # Evaluate baseline
    metrics = evaluate_baseline(
        test_loader=dataloader,
        device=args.device,
        num_samples=args.num_samples,
        lightning_threshold=args.lightning_threshold
    )

    # Print results
    print("\nEvaluation Results:")
    for steps, step_metrics in metrics.items():
        print(f"\nSteps: {steps}")
        for metric_name, value in step_metrics.items():
            print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()