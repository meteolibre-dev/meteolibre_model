import torch
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from skimage.metrics import mean_squared_error as mse_skimage  # If skimage available; else torch MSE
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from meteolibre_model.diffusion.rectified_flow_lightning_shortcut import full_image_generation, denormalize

def generate_data(model, test_loader, device, num_steps_list=[8, 16, 128], num_samples=10):
    """
    Generate all data for evaluation.
    
    Args:
        model: Trained model.
        test_loader: DataLoader for test set.
        device: 'cuda' or 'cpu'.
        num_steps_list: Inference budgets to test.
        num_samples: Samples per input for probabilistic metrics.
    
    Returns:
        all_gens: Dict of generated data per step budget.
        all_gts: List of ground truths.
    """
    model.eval()
    
    # Initialize storage for generated data and ground truths
    all_gens = {steps: [] for steps in num_steps_list}  # List of (mean_sat, mean_light) per batch
    all_gts = []  # List of (gt_sat, gt_light) per batch
    
    with torch.no_grad():
        for batch in test_loader:
            sat_data = batch["sat_patch_data"].permute(0, 2, 1, 3, 4)
            lightning_data = batch["lightning_patch_data"].permute(0, 2, 1, 3, 4)
            c_sat = sat_data.shape[1]
            c_lightning = lightning_data.shape[1]
            
            # Ground truth (denormalized)
            gt_sat, gt_light = denormalize(sat_data[:, :, 4:], lightning_data[:, :, 4:], device)
            all_gts.append((gt_sat, gt_light))
            
            for steps in num_steps_list:
                gens_sat, gens_light = [], []
                for _ in range(num_samples):  # Multi-sample
                    gen, _ = full_image_generation(model, batch, num_steps=steps, device=device)
                    sat_gen, light_gen = denormalize(gen[:, :c_sat], gen[:, c_lightning:], device)
                    gens_sat.append(sat_gen)
                    gens_light.append(light_gen)
                
                # Mean gen for deterministic metrics
                mean_sat = torch.mean(torch.stack(gens_sat), dim=0)
                mean_light = torch.mean(torch.stack(gens_light), dim=0)
                all_gens[steps].append((mean_sat, mean_light))
    
    return all_gens, all_gts


def compute_metrics(all_gens, all_gts, device, num_steps_list=[8, 16, 128], lightning_threshold=0.05):
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
                       'light_mae': [], 'light_pod': [], 'light_far': [], 'light_csi': []} 
               for steps in num_steps_list}
    
    psnr = PeakSignalNoiseRatio(data_range=(-8, 8.0)).to(device)  # Adjust range based on denorm data
    ssim = StructuralSimilarityIndexMeasure(data_range=(-8, 8.0)).to(device)
    
    for batch_idx in range(len(all_gts)):
        gt_sat, gt_light = all_gts[batch_idx]
        for steps in num_steps_list:
            mean_sat, mean_light = all_gens[steps][batch_idx]
            
            # Sat metrics
            mse_val = mse_skimage(mean_sat.cpu().numpy(), gt_sat.cpu().numpy())
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
            prec, rec, f1, _ = precision_recall_fscore_support(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
            hits = np.sum((pred_bin == 1) & (gt_bin == 1))
            misses = np.sum((pred_bin == 0) & (gt_bin == 1))
            falses = np.sum((pred_bin == 1) & (gt_bin == 0))
            pod = hits / (hits + misses) if (hits + misses) > 0 else 0
            far = falses / (hits + falses) if (hits + falses) > 0 else 0
            csi = hits / (hits + misses + falses) if (hits + misses + falses) > 0 else 0
            metrics[steps]['light_pod'].append(pod)
            metrics[steps]['light_far'].append(far)
            metrics[steps]['light_csi'].append(csi)
    
    # Average metrics
    for steps in num_steps_list:
        for k in metrics[steps]:
            metrics[steps][k] = np.mean(metrics[steps][k])
    
    return metrics


def evaluate_model(model, test_loader, device, num_steps_list=[8, 16, 128], num_samples=10, lightning_threshold=0.05):
    """
    Evaluate model on test set for different step budgets.
    
    Args:
        model: Trained model.
        test_loader: DataLoader for test set.
        device: 'cuda' or 'cpu'.
        num_steps_list: Inference budgets to test.
        num_samples: Samples per input for probabilistic metrics.
        lightning_threshold: Binarize lightning > this as 'event'.
    
    Returns:
        Dict of metrics per step budget.
    """
    # Step 1: Generate all data
    all_gens, all_gts = generate_data(model, test_loader, device, num_steps_list, num_samples)
    
    # Step 2: Compute metrics
    metrics = compute_metrics(all_gens, all_gts, device, num_steps_list, lightning_threshold)
    
    return metrics

# Usage: results = evaluate_model(model, test_loader, device)
# print(results)  # e.g., {1: {'sat_mse': 0.05, ...}, ...}