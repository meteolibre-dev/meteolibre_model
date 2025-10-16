from tqdm import tqdm

import torch
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from meteolibre_model.diffusion.rectified_flow_lightning import full_image_generation, normalize

def generate_data(model, test_loader, device, num_steps_list=[128], num_samples=10):
    """
    Generate all data for evaluation.
    
    Args:
        model: Trained model.
        test_loader: DataLoader for test set.
        device: 'cuda' or 'cpu'.
        num_steps_list: Inference budgets to test.
        num_samples: Number of batches to evaluate.
    
    Returns:
        all_gens: Dict of generated data per step budget.
        all_gts: List of ground truths.
    """
    model.eval()
    
    # Initialize storage for generated data and ground truths
    all_gens = {steps: [] for steps in num_steps_list}
    all_gts = []
    
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_samples:
                break

            print(f"Begin Eval on batch {batch_idx + 1}/{num_samples}")

            batch["sat_patch_data"] = batch["sat_patch_data"].to(device)
            batch["lightning_patch_data"] = batch["lightning_patch_data"].to(device)
            batch["spatial_position"] = batch["spatial_position"].to(device)

            sat_data = batch["sat_patch_data"].permute(0, 2, 1, 3, 4)
            lightning_data = batch["lightning_patch_data"].permute(0, 2, 1, 3, 4)

            c_sat = sat_data.shape[1]
            c_lightning = lightning_data.shape[1]
            
            # Ground truth (with normalized)
            gt_sat, gt_light = normalize(sat_data[:, :, 4:], lightning_data[:, :, 4:], device)
            all_gts.append((gt_sat.squeeze(2), gt_light.squeeze(2)))
            
            for steps in num_steps_list:
                gen, _ = full_image_generation(model, batch, steps=steps, device=device, nb_element=test_loader.batch_size)
                sat_gen, light_gen = gen[:, :c_sat].to(device), gen[:, c_sat:].to(device)
                
                all_gens[steps].append((sat_gen.squeeze(2), light_gen.squeeze(2)))
    
    return all_gens, all_gts


def compute_metrics(all_gens, all_gts, device, num_steps_list=[128], lightning_threshold=0.05):
    """
    Compute metrics from generated data and ground truths.
    
    Args:
        all_gens: Dict of generated data per step budget.
        all_gts: List of ground truths.
        device: 'cuda' or 'cpu'.
        num_steps_list: Inference budgets to test.
        lightning_threshold: Binarize lightning > this as 'event'.
    
    Returns:
        Dict of metrics per step budget.
    """
    # Initialize metrics storage
    metrics = {steps: {'sat_mse': [], 'sat_psnr': [], 'sat_ssim': [], 
                       'light_mae': [], 'light_precision': [], 'light_recall': [], 'light_f1': []} 
               for steps in num_steps_list}
    
    psnr = PeakSignalNoiseRatio(data_range=(-4, 4.0)).to(device)  # Adjust range based on denorm data
    ssim = StructuralSimilarityIndexMeasure(data_range=(-4, 4.0)).to(device)
    
    for batch_idx in range(len(all_gts)):
        gt_sat, gt_light = all_gts[batch_idx]
        for steps in num_steps_list:
            mean_sat, mean_light = all_gens[steps][batch_idx]

            # Sat metrics
            mse_val = torch.nn.functional.mse_loss(mean_sat.cpu(), gt_sat.cpu())
            psnr_val = psnr(mean_sat, gt_sat)
            ssim_val = ssim(mean_sat, gt_sat)
            metrics[steps]['sat_mse'].append(mse_val)
            metrics[steps]['sat_psnr'].append(psnr_val.item())
            metrics[steps]['sat_ssim'].append(ssim_val.item())
            
            # Lightning metrics (assume per-grid regression)
            mae_val = torch.mean(torch.abs(mean_light - gt_light)).item()
            metrics[steps]['light_mae'].append(mae_val)
            
            # Binarize for event metrics
            pred_bin = (mean_light > lightning_threshold).float().cpu().numpy()
            gt_bin = (gt_light > lightning_threshold).float().cpu().numpy()

            prec, rec, f1, _ = precision_recall_fscore_support(gt_bin.flatten(), pred_bin.flatten(), average='binary', zero_division='warn')

            metrics[steps]['light_precision'].append(prec)
            metrics[steps]['light_recall'].append(rec)
            metrics[steps]['light_f1'].append(f1)
    
    # Average metrics
    avg_metrics = {}
    for steps in num_steps_list:
        avg_metrics[steps] = {}
        for k in metrics[steps]:
            avg_metrics[steps][k] = np.mean(metrics[steps][k])
    
    return avg_metrics


def generate_baseline_data(test_loader, device, num_samples=10):
    """
    Generate baseline data using previous ground truth data.
    
    Args:
        test_loader: DataLoader for test set.
        device: 'cuda' or 'cpu'.
        num_samples: Number of batches to evaluate.
    
    Returns:
        all_gens: Dict with 'baseline' key containing generated data.
        all_gts: List of ground truths.
    """
    # Initialize storage for generated data and ground truths
    all_gens = {'baseline': []}
    all_gts = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_samples:
                break

            print(f"Begin Baseline Eval on batch {batch_idx + 1}/{num_samples}")

            batch["sat_patch_data"] = batch["sat_patch_data"].to(device)
            batch["lightning_patch_data"] = batch["lightning_patch_data"].to(device)
            batch["spatial_position"] = batch["spatial_position"].to(device)

            sat_data = batch["sat_patch_data"].permute(0, 2, 1, 3, 4)
            lightning_data = batch["lightning_patch_data"].permute(0, 2, 1, 3, 4)

            # Ground truth (with normalized)
            gt_sat, gt_light = normalize(sat_data[:, :, 4:], lightning_data[:, :, 4:], device)
            all_gts.append((gt_sat.squeeze(2), gt_light.squeeze(2)))
            
            # Baseline: use previous time step (index 3)
            sat_gen, light_gen = normalize(sat_data[:, :, 3:4], lightning_data[:, :, 3:4], device)
            all_gens['baseline'].append((sat_gen.squeeze(2), light_gen.squeeze(2)))
    
    return all_gens, all_gts


def evaluate_baseline(test_loader, device, num_samples=10, lightning_threshold=0.05):
    """
    Evaluate baseline model on test set.
    
    Args:
        test_loader: DataLoader for test set.
        device: 'cuda' or 'cpu'.
        num_samples: Number of batches to evaluate.
        lightning_threshold: Binarize lightning > this as 'event'.
    
    Returns:
        Dict of metrics for baseline.
    """
    # Step 1: Generate all data
    all_gens, all_gts = generate_baseline_data(test_loader, device, num_samples)

    print("Baseline generation finished")
    
    # Step 2: Compute metrics
    metrics = compute_metrics(all_gens, all_gts, device, num_steps_list=['baseline'], lightning_threshold=lightning_threshold)

    print("Computing baseline metrics finished")
    
    return metrics


def evaluate_model(model, test_loader, device, num_steps_list=[128], num_samples=10, lightning_threshold=0.05):
    """
    Evaluate model on test set for different step budgets.
    
    Args:
        model: Trained model.
        test_loader: DataLoader for test set.
        device: 'cuda' or 'cpu'.
        num_steps_list: Inference budgets to test.
        num_samples: Number of batches to evaluate.
        lightning_threshold: Binarize lightning > this as 'event'.
    
    Returns:
        Dict of metrics per step budget.
    """
    # Step 1: Generate all data
    all_gens, all_gts = generate_data(model, test_loader, device, num_steps_list, num_samples)

    print("Generation finished")
    
    # Step 2: Compute metrics
    metrics = compute_metrics(all_gens, all_gts, device, num_steps_list, lightning_threshold)

    print("Computing metrics finished")
    
    return metrics

# Usage: results = evaluate_model(model, test_loader, device)
# print(results)  # e.g., {1: {'sat_mse': 0.05, ...}, ...}