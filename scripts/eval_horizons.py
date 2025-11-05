import argparse
import os
import sys
from datetime import datetime
import numpy as np
import yaml

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Load config
config_path = os.path.join(project_root, "meteolibre_model/config/configs.yml")
with open(config_path) as f:
    config = yaml.safe_load(f)
params = config["model_v1_mtg_world_lightning_shortcut"]

from meteolibre_model.evaluate.horizon_evaluation import quick_evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model performance at various forecasting horizons using tiled inference."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Path to the pre-trained model .safetensors file (required unless --baseline is used).",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the HDF5 file containing the initial context and ground truth frames.",
    )
    parser.add_argument(
        "--initial_date_str",
        type=str,
        required=True,
        help="Initial date for the first forecast (horizon 1) in format 'YYYY-MM-DD HH:MM' (e.g., '2025-10-14 04:00').",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs='+',
        default=[1, 2, 3, 6, 12, 18],
        help="List of forecast horizons to evaluate (e.g., 1 3 6 12).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu).",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=128,
        help="Size of patches for tiled inference.",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        default=8,
        help="Number of denoising steps for each tiled inference.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for processing patches during inference.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (NPZ file).",
    )
    parser.add_argument(
        "--context_frames",
        type=int,
        default=params["model"]["context_frames"],
        help="Number of context frames expected by the model.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Compute persistence baseline metrics instead of model evaluation.",
    )
    args = parser.parse_args()

    if args.baseline:
        print(f"Computing baseline persistence at horizons: {args.horizons}")
    else:
        print(f"Evaluating model at horizons: {args.horizons}")
    print(f"Data file: {args.data_file}")
    print(f"Initial date: {args.initial_date_str}")
    if not args.baseline:
        print(f"Device: {args.device}")
        print(f"Model path: {args.model_path}")

    # Run evaluation
    results = quick_evaluate(
        model_path=args.model_path,
        data_file=args.data_file,
        initial_date_str=args.initial_date_str,
        horizons=args.horizons,
        device=args.device,
        patch_size=args.patch_size,
        denoising_steps=args.denoising_steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        baseline=args.baseline,
    )

    # Print results
    print("\nEvaluation Results:")
    for horizon, metrics in results.items():
        print(f"\nHorizon {horizon}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")

    print(f"\nResults saved to {args.output_dir}/evaluation_results.npz")


if __name__ == "__main__":
    import torch  # For device check
    main()
