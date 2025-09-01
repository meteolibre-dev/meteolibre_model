"""
Score-based generative model implementation for weather forecasting.
This module provides functions for training and generation using score-based diffusion.
"""

import torch
import einops
from tqdm import tqdm

from meteolibre_model.diffusion.utils import MEAN_CHANNEL, STD_CHANNEL

# -- Parameters --
CLIP_MIN = -4


SIGMA_DATA = 1.0

def normalize(batch_data, device):
    """
    Normalize the batch data using precomputed mean and std.
    """
    batch_data = (
        (
            batch_data
            - MEAN_CHANNEL.unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .to(device)
        )
        / STD_CHANNEL.unsqueeze(0)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .to(device)
    )

    # Clamp to prevent extreme values
    batch_data = batch_data.clamp(CLIP_MIN, 4)

    return batch_data

def trainer_step_edm_preconditioned_loss(model, batch, device):
    """
    Performs a single training step using the Karras et al. preconditioning.
    This is the recommended, numerically stable approach.
    """
    with torch.no_grad():
        # Data preparation (same as before)
        batch_data = batch["patch_data"].permute(0, 2, 1, 3, 4)
        b, c, t_frames, h, w = batch_data.shape
        batch_data = normalize(batch_data, device)
        x_context = batch_data[:, :, :4]
        x_target = batch_data[:, :, 4:] # This is our clean data, x_0

        last_context_frame = x_context[:, :, 3:4] # Shape: (B, C, 1, H, W)
        x_target_residual = x_target - last_context_frame

        mask_data = x_target != CLIP_MIN

        # Sample sigma (same as before)
        P_mean = -1.2
        P_std = 1.5
        log_sigma = P_mean + P_std * torch.randn(b, device=device)
        sigma = torch.exp(log_sigma)
        sigma_sq = sigma.pow(2)

        # Generate noise
        noise = torch.randn_like(x_target)
        
        # Reshape sigma for broadcasting
        sigma_exp = sigma.view(b, 1, 1, 1, 1)
        
        # Add noise: x_t = x_target + sigma * noise
        x_t = x_target_residual + sigma_exp * noise

    # --- NEW: Preconditioning ---
    # These are the scaling factors from the EDM paper (Appendix B.2)
    c_skip = SIGMA_DATA**2 / (sigma_sq + SIGMA_DATA**2)
    c_out = sigma * SIGMA_DATA / (sigma_sq + SIGMA_DATA**2).sqrt()
    c_in = 1 / (sigma_sq + SIGMA_DATA**2).sqrt()
    
    # Reshape for broadcasting
    c_skip = c_skip.view(b, 1, 1, 1, 1)
    c_out = c_out.view(b, 1, 1, 1, 1)
    c_in = c_in.view(b, 1, 1, 1, 1)
    
    # 1. Scale the model input
    model_input_scaled = c_in * x_t

    # Model input: concatenate context and the scaled x_t
    model_input = torch.cat([x_context, model_input_scaled], dim=2)

    # Condition on log(sigma)
    context_info = batch["spatial_position"]
    context_global = torch.cat([context_info, log_sigma.unsqueeze(1)], dim=1)

    # --- Model now predicts the denoised data x_0 ---
    # The network's output is F_theta
    F_theta = model(model_input.float(), context_global.float())
    
    # 2. Un-scale the model output to get the final denoised prediction
    denoised_prediction = c_skip * x_t + c_out * F_theta

    # --- New Loss Calculation ---
    # The loss weight is now simpler: just 1 / (c_out^2)
    loss_weight = (sigma_sq + SIGMA_DATA**2) / (sigma_sq * SIGMA_DATA**2) # This is the same as before
    loss_weight = loss_weight.clamp(min=0, max=100)
    loss_weight_exp = loss_weight.view(b, 1, 1, 1, 1)

    # The loss is the weighted MSE between the prediction and the *original clean data*
    squared_error = (denoised_prediction - x_target)**2
    weighted_squared_error = loss_weight_exp * squared_error
    
    loss = torch.mean(weighted_squared_error[mask_data])
    
    return loss
def edm_sampler_preconditioned(model, batch, x_context, num_steps=100, device="cuda", parametrization="standard",
                               sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """
    Corrected sampler for the preconditioned model.
    """
    model.eval()
    with torch.no_grad():
        model.to(device)
        x_context = normalize(x_context[[0]], device)
        last_context_frame = x_context[:, :, 3:4]
        context_info = batch["spatial_position"].to(device)[[0]]
        b, c, _, h, w = x_context.shape

        # Define the sigma schedule (same as before)
        step_indices = torch.arange(num_steps, device=device)
        t_steps = (sigma_max**(1/rho) + step_indices / (num_steps - 1) * (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
        sigmas = torch.cat([t_steps, torch.tensor([0.], device=device)])

        # Start with pure noise (same as before)
        x_t = torch.randn(b, c, 2, h, w, device=device) * sigmas[0]

        # Main sampling loop
        for i in range(num_steps):
            sigma_t = sigmas[i]
            sigma_next = sigmas[i+1]
            sigma_t_sq = sigma_t.pow(2)

            # --- NEW: Calculate Preconditioning Constants ---
            c_skip = SIGMA_DATA**2 / (sigma_t_sq + SIGMA_DATA**2)
            c_out = sigma_t * SIGMA_DATA / (sigma_t_sq + SIGMA_DATA**2).sqrt()
            c_in = 1 / (sigma_t_sq + SIGMA_DATA**2).sqrt()

            # --- CHANGED: Scale input and call model ---
            model_input_scaled = c_in * x_t # <-- Scale the input
            model_input = torch.cat([x_context, model_input_scaled], dim=2)
            
            log_sigma_t = torch.log(sigma_t).expand(b)
            context_global = torch.cat([context_info, log_sigma_t.unsqueeze(1)], dim=1)

            # Model predicts F_theta
            F_theta = model(model_input.float(), context_global.float())
            
            # --- CHANGED: Calculate denoised estimate with preconditioning ---
            denoised_estimate = c_skip * x_t + c_out * F_theta # <-- Un-scale the output

            # Euler step (this part remains the same)
            d = (x_t - denoised_estimate) / sigma_t
            x_next = x_t + d * (sigma_next - sigma_t)
            
            x_t = x_next
        
        # Final output processing (same as before)
        x_t_absolute = x_t + last_context_frame.expand(-1, -1, 2, -1, -1)


    model.train()
    return x_t.cpu()


def edm_sampler_heun(
    model, batch, x_context, num_steps=35, device="cuda",
    sigma_min=0.002, sigma_max=80.0, rho=7.0
):
    """
    Generates samples using the 2nd order Heun's method sampler from Karras et al. 2022.
    This corresponds to Algorithm 1 in the paper.
    """
    model.eval()
    with torch.no_grad():
        model.to(device)
        x_context = normalize(x_context[[0]], device)
        last_context_frame = x_context[:, :, 3:4]
        context_info = batch["spatial_position"].to(device)[[0]]
        b, c, _, h, w = x_context.shape

        # Define the sigma schedule as per Equation 5 in the paper [cite: 170]
        step_indices = torch.arange(num_steps, device=device)
        t_steps = (
            sigma_max**(1/rho) + step_indices / (num_steps - 1) * (sigma_min**(1/rho) - sigma_max**(1/rho))
        )**rho
        sigmas = torch.cat([t_steps, torch.tensor([0.], device=device)])

        # Start with pure noise, scaled by the first sigma
        x_t = torch.randn(b, c, 2, h, w, device=device) * sigmas[0]

        # Main sampling loop (using tqdm for progress)
        for i in tqdm(range(num_steps), disable=False):
            sigma_t = sigmas[i]
            sigma_next = sigmas[i+1]
            
            # --- 1. Predictor step (Euler step from Algorithm 1, lines 4-5) ---
            
            # Get the derivative (d_i in the paper)
            # This is dx/dt = (x - D(x, t)) / t, with t=sigma_t
            # First, we need the denoiser output D(x, t)
            
            sigma_t_sq = sigma_t.pow(2)
            c_skip = SIGMA_DATA**2 / (sigma_t_sq + SIGMA_DATA**2)
            c_out = sigma_t * SIGMA_DATA / (sigma_t_sq + SIGMA_DATA**2).sqrt()
            c_in = 1 / (sigma_t_sq + SIGMA_DATA**2).sqrt()

            model_input_scaled = c_in * x_t
            model_input = torch.cat([x_context, model_input_scaled], dim=2)
            log_sigma_t = torch.log(sigma_t).expand(b)
            context_global = torch.cat([context_info, log_sigma_t.unsqueeze(1)], dim=1)
            
            F_theta = model(model_input.float(), context_global.float())
            denoised_estimate = c_skip * x_t + c_out * F_theta
            
            # This is d_i in Algorithm 1
            d_i = (x_t - denoised_estimate) / sigma_t
            
            # Take the Euler step
            x_next_predictor = x_t + d_i * (sigma_next - sigma_t)
            
            # --- 2. Corrector step (Algorithm 1, lines 7-8) ---
            
            # Only apply correction if the next sigma is not zero
            if sigma_next != 0:
                # Get the derivative at the predicted next point (d'_i in the paper)
                sigma_next_sq = sigma_next.pow(2)
                c_skip_next = SIGMA_DATA**2 / (sigma_next_sq + SIGMA_DATA**2)
                c_out_next = sigma_next * SIGMA_DATA / (sigma_next_sq + SIGMA_DATA**2).sqrt()
                c_in_next = 1 / (sigma_next_sq + SIGMA_DATA**2).sqrt()

                model_input_scaled_next = c_in_next * x_next_predictor
                model_input_next = torch.cat([x_context, model_input_scaled_next], dim=2)
                log_sigma_next = torch.log(sigma_next).expand(b)
                context_global_next = torch.cat([context_info, log_sigma_next.unsqueeze(1)], dim=1)

                F_theta_next = model(model_input_next.float(), context_global_next.float())
                denoised_estimate_next = c_skip_next * x_next_predictor + c_out_next * F_theta_next
                
                # This is d'_i in Algorithm 1
                d_prime_i = (x_next_predictor - denoised_estimate_next) / sigma_next
                
                # Take the final Heun step using the average of the two derivatives
                x_t = x_t + (d_i + d_prime_i) / 2.0 * (sigma_next - sigma_t)
            else:
                # For the final step to sigma=0, just use the Euler prediction
                x_t = x_next_predictor

        # Final output processing
        x_t_absolute = x_t + last_context_frame.expand(-1, -1, 2, -1, -1)

    model.train()
    return x_t_absolute.cpu() # Return the absolute values, not the residual