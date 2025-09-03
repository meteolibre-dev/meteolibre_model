import torch
import numpy as np
import einops
import argparse
from tqdm import tqdm
import os
import sys

# Add project root to sys.path
project_root = os.path.abspath("/home/adrienbufort/Documents/workspace/meteolibre_model/")
sys.path.insert(0, project_root)

from meteolibre_model.models.dc_3dunet_film import UNet_DCAE_3D
from meteolibre_model.diffusion.rectified_flow import (
    full_image_generation as rectified_flow_generate,
    normalize,
    COEF_NOISE,
    get_proper_noise,
    ds_dt,
)
from safetensors.torch import load_file

def _extract_patch(image, x, y, patch_size):
    return image[..., y : y + patch_size, x : x + patch_size]

def _place_patch(full_image, patch, x, y, patch_size):
    full_image[..., y : y + patch_size, x : x + patch_size] = patch
    return full_image

def tiled_inference(
    model,
    initial_context, # (B, C, 4, H_small, W_small)
    big_image_dims, # (H_big, W_big)
    patch_size, # H_small or W_small
    overlap,
    steps,
    device,
    parametrization="standard",
    nb_channels=12,
    normalize_func=normalize,
):
    model.eval()
    model.to(device)

    # Initial context (first 4 frames) should be resized to the big image dimensions
    # For now, let's assume it's just repeated or upsampled. In a real scenario, you'd
    # have a way to obtain this context at high resolution.
    # For this exercise, we'll create a dummy high-res context by repeating the small one.
    # This needs to be adapted based on how you get your actual initial context.
    _, C, T_ctx, H_small, W_small = initial_context.shape
    H_big, W_big = big_image_dims
    
    # Simple upsampling for context - replace with a proper method if available
    # Or, if context is also tiled, it would be handled differently.
    # For now, let's assume `initial_context` is already high-res or upsampled within this function
    # if it's coming from a small patch.
    # For simplicity, let's just make it a noisy large image. The user wants to start with a noisy 3000x2500 image

    # Initialize x_t (noisy image) for the full large image
    # (B, C, 2, H_big, W_big)
    x_t_full_res = torch.randn(1, nb_channels, 2, H_big, W_big, device=device) # Start with noise for the target 2 frames

    # Create dummy spatial_position for the full image, assuming a constant value for now
    # This needs to be carefully handled in a real-world scenario, as spatial_position
    # might vary across the large image. For simplicity, we'll use a placeholder.
    # The original context_info has a spatial_position of shape (1, 3).
    # We will use a dummy one for now.
    dummy_spatial_position = torch.tensor([[0.0, 0.0, 0.0]], device=device) # (1, 3)

    overlap_pixels = int(patch_size * overlap)
    stride = patch_size - overlap_pixels
    
    # Ensure stride is at least 1 to avoid infinite loops or no progression
    if stride <= 0:
        raise ValueError("Overlap is too large, resulting in non-positive stride. Adjust patch_size or overlap.")

    # Calculate sampling grid dynamically to cover the full image
    y_starts = list(range(0, H_big - patch_size + 1, stride))
    if (H_big - patch_size) % stride != 0:
        y_starts.append(H_big - patch_size) # Ensure last patch covers the bottom edge
    
    x_starts = list(range(0, W_big - patch_size + 1, stride))
    if (W_big - patch_size) % stride != 0:
        x_starts.append(W_big - patch_size) # Ensure last patch covers the right edge

    dt = 1.0 / steps

    for i in tqdm(range(steps), desc="Tiled Denoising"):
        t_val = 1 - i * dt
        t_next_val = 1 - (i + 1) * dt
        t_batch_val = torch.full((1,), t_val, device=device) # (1,)
        t_next_batch_val = torch.full((1,), t_next_val, device=device) # (1,)

        s_t, logsnr_t = get_proper_noise(COEF_NOISE, t_batch_val)
        s_t_next, logsnr_t_next = get_proper_noise(COEF_NOISE, t_next_batch_val)

        t_expanded_full = t_batch_val.view(1, 1, 1, 1, 1).expand(1, nb_channels, 2, patch_size, patch_size)
        derivative = ds_dt(t_expanded_full) # Derivative assumes a specific shape, needs to be flexible

        aggregated_velocity = torch.zeros_like(x_t_full_res, device=device)
        overlap_counts = torch.zeros_like(x_t_full_res, device=device)

        for y_start in y_starts:
            for x_start in x_starts:
                # Extract patch from the current full noisy image
                patch_x_t = _extract_patch(x_t_full_res, x_start, y_start, patch_size)
                
                # Assume initial_context is the context for all patches for this step
                # In a real scenario, initial_context might also be tiled or derived per patch
                # For autoregressive part, initial_context will be the *last 4 actual forecast frames*
                # For now, we take the initial_context provided (which is fixed here)
                
                # --- Prepare context_global for the model ---
                # context_info: (1, 3)
                # logsnr_t: (1,)
                # We need context_global of shape (1, 4)
                context_global = torch.cat([dummy_spatial_position, logsnr_t.unsqueeze(-1) / 10.], dim=1) # (1, 4)

                # Model input: concatenate context and x_t patch
                # x_context: (1, C, 4, patch_size, patch_size)
                # patch_x_t: (1, C, 2, patch_size, patch_size)
                model_input = torch.cat([initial_context, patch_x_t], dim=2) # (1, C, 6, patch_size, patch_size)
                
                # Predict velocity (v-prediction)
                pred_patch = model(model_input.float(), context_global.float())[:, :, 4:, :, :] # (1, C, 2, patch_size, patch_size)


                v_patch = pred_patch
                
                aggregated_velocity[..., y_start : y_start + patch_size, x_start : x_start + patch_size] += v_patch
                overlap_counts[..., y_start : y_start + patch_size, x_start : x_start + patch_size] += 1

        # Average overlapping velocities
        # Avoid division by zero for non-covered regions (though with proper tiling this shouldn't happen)
        overlap_counts[overlap_counts == 0] = 1 
        averaged_velocity = aggregated_velocity / overlap_counts

        # Euler prediction for Heun's
        x_euler = x_t_full_res - dt * averaged_velocity

        # Heun's method for next step - requires predicting at t_next_val and x_euler
        # This part assumes that the model can still produce an endpoint prediction
        # based on an estimated x_euler,
        # For simplicity, we'll re-run patches with x_euler to get pred_next_patch
        # and then combine for v2. This is computationally expensive but conceptually direct.
        
        aggregated_pred_next = torch.zeros_like(x_t_full_res, device=device)
        
        context_global_next = torch.cat([dummy_spatial_position, logsnr_t_next.unsqueeze(-1) / 10.], dim=1) # (1, 4)

        for y_start in y_starts:
            for x_start in x_starts:
                patch_x_euler = _extract_patch(x_euler, x_start, y_start, patch_size)
                model_input_next_patch = torch.cat([initial_context, patch_x_euler], dim=2)
                pred_next_patch = model(model_input_next_patch.float(), context_global_next.float())[:, :, 4:, :, :]
                aggregated_pred_next[..., y_start : y_start + patch_size, x_start : x_start + patch_size] += pred_next_patch

        averaged_pred_next = aggregated_pred_next / overlap_counts # Reuse overlap_counts

        v2 = averaged_pred_next

        # Heun's step: x_{t-dt} = x_t - (dt/2) * (v1 + v2)
        x_t_full_res = x_t_full_res - (dt / 2) * (averaged_velocity + v2)
        x_t_full_res = x_t_full_res.clamp(-7, 7) # Clamp to prevent divergence

    # Always add back the last context since always forecasting residual
    # Assuming initial_context is already normalized and the model expects residual.
    # The initial_context is (B, C, 4, H_small, W_small). 
    # The last frame of context is initial_context[:, :, 3:4].
    # This needs to be upsampled to H_big, W_big if initial_context itself is small.
    # If initial_context is truly the context for the *entire* large image, 
    # then initial_context[:, :, 3:4] needs to be (1, nb_channels, 1, H_big, W_big).

    # For this script, let's assume `initial_context` is already large, 
    # matching `H_big, W_big` for its spatial dimensions,
    # and has been provided correctly in the first place or upscaled before calling `tiled_inference`.
    # Let's adjust `initial_context` to reflect it is already upsampled for the first part of the script.
    
    # We need to construct a proper initial_context (B, C, 4, H_big, W_big)
    # The user states: "initially generate a noisy 3000x2500 image" meaning the target.
    # The context would be separate. If the context is also provided initially at small size,
    # then it needs to be upsampled or processed in a tiled manner as well.
    # For now, let's create a *dummy high-res initial_context*
    
    # Create the high res context (B, C, 4, H_big, W_big)
    dummy_high_res_context = torch.randn(1, nb_channels, 4, H_big, W_big, device=device)
    # Normailize this dummy context as well
    dummy_high_res_context = normalize_func(dummy_high_res_context, device)

    last_context_frame = dummy_high_res_context[:, :, 3:4] # (1, C, 1, H_big, W_big)
    
    # Expand it to match the 2-frame generated output
    last_context_frame_expanded = last_context_frame.expand(-1, -1, 2, -1, -1)
    
    x_t_full_res = x_t_full_res + last_context_frame_expanded

    model.train()
    return x_t_full_res.cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Tiled Diffusion for large image forecasting with Rectified Flow."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/adrienbufort/Documents/workspace/meteolibre_model/models/epoch_11.safetensors",
        help="Path to the pre-trained model .safetensors file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_forecasts",
        help="Directory to save generated forecasts.",
    )
    parser.add_argument(
        "--forecast_steps",
        type=int,
        default=60,
        help="Number of autoregressive forecast steps to generate (total frames).",
    )
    parser.add_argument(
        "--target_H", type=int, default=2500, help="Target height for the large image."
    )
    parser.add_argument(
        "--target_W", type=int, default=3000, help="Target width for the large image."
    )
    parser.add_argument(
        "--patch_size", type=int, default=128, help="Size of patches trained on (e.g., 128 for 128x128)."
    )
    parser.add_argument(
        "--overlap_ratio",
        type=float,
        default=0.25,
        help="Overlap ratio between patches (e.g., 0.25 for 25% overlap).",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        default=100,
        help="Number of denoising steps for each tiled diffusion process.",
    )
    parser.add_argument(
        "--model_channels",
        type=int,
        default=12,
        help="Number of input/output channels for the model.",
    )
    parser.add_argument(
        "--model_features",
        nargs="+",
        type=int,
        default=[32, 64, 128, 256], #[64, 128, 256],
        help="List of feature counts for the U-Net architecture.",
    )
    parser.add_argument(
        "--context_dim",
        type=int,
        default=4,
        help="Dimension of the context for FiLM conditioning.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Dimension of the embedding for FiLM conditioning.",
    )
    parser.add_argument(
        "--context_frames",
        type=int,
        default=4,
        help="Number of context frames expected by the model.",
    )
    parser.add_argument(
        "--parametrization",
        type=str,
        default="standard",
        choices=["standard", "endpoint"],
        help="Parametrization used in the Rectified Flow model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    model = UNet_DCAE_3D(
        in_channels=args.model_channels,
        out_channels=args.model_channels,
        features=args.model_features,
        context_dim=args.context_dim,
        embedding_dim=args.embedding_dim,
        context_frames=args.context_frames,
    )
    
    # Load model weights
    if os.path.exists(args.model_path):
        loaded_state_dict = load_file(args.model_path)
        model.load_state_dict(loaded_state_dict)
        print(f"Loaded model weights from {args.model_path}")
    else:
        print(f"Warning: Model weights not found at {args.model_path}. Using randomly initialized model.")

    model.to(args.device)

    # --- Autoregressive Generation Loop ---
    
    # Initial context (first 4 frames) for the *large image*
    # For now, generate random data for simplicity. In a real application,
    # you would load your actual initial observations (e.g., 4 frames of 3000x2500 data)
    # and normalize them.
    # Shape: (B, C, T_ctx, H_big, W_big) -> (1, args.model_channels, 4, args.target_H, args.target_W)
    current_high_res_context = torch.randn(
        1, args.model_channels, args.context_frames, args.target_H, args.target_W, device=args.device
    )
    current_high_res_context = normalize(current_high_res_context, args.device) # Normalize initial true data

    # Store all generated frames
    all_generated_frames = [current_high_res_context.cpu()] # Store initial context

    # Each step of autoregressive generation predicts 2 frames.
    # We need to run (forecast_steps / 2) iterations.
    for step_idx in range(args.forecast_steps // 2):
        print(f"Autoregressive step {step_idx + 1}/{args.forecast_steps // 2}")

        # Perform tiled inference for the next 2 frames
        # The `initial_context` for `tiled_inference` is now `current_high_res_context`
        # which acts as the fixed conditioning input for predicting the *next 2 frames*.
        generated_two_frames = tiled_inference(
            model=model,
            initial_context=current_high_res_context,
            big_image_dims=(args.target_H, args.target_W),
            patch_size=args.patch_size,
            overlap=args.overlap_ratio,
            steps=args.denoising_steps,
            device=args.device,
            parametrization=args.parametrization,
            nb_channels=args.model_channels,
            normalize_func=normalize,
        ) # Shape: (1, C, 2, H_big, W_big)
        
        all_generated_frames.append(generated_two_frames.cpu())

        # Update current_high_res_context for the next autoregressive step
        # Shift the context: remove the oldest 2 frames, add the newly generated 2 frames
        current_high_res_context = torch.cat(
            [current_high_res_context[:, :, 2:, :, :], generated_two_frames.to(args.device)], dim=2
        ) # (1, C, 4, H_big, W_big)

        # Optional: Save intermediate forecast or visualize (e.g., save every 10 steps)
        # For simplicity, we'll save the final full forecast.
    
    # Concatenate all generated frames into a single tensor
    # The first item in all_generated_frames has 4 frames, subsequent have 2.
    final_forecast = torch.cat(all_generated_frames, dim=2) # (1, C, 4 + forecast_steps, H_big, W_big)

    # Example: Save the first channel of the first (and only) batch item to a numpy file
    # for visualization or further processing.
    # Or, save each frame as an image if visualization is needed directly here.
    output_filepath = os.path.join(args.output_dir, "full_forecast.npy")
    np.save(output_filepath, final_forecast.squeeze(0).numpy())
    print(f"Full forecast saved to {output_filepath}")

    print("Tiled diffusion and autoregressive forecasting complete.")


if __name__ == "__main__":
    main()
