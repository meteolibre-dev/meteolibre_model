"""
Rectified Flow implementation for weather forecasting diffusion model.
This module provides functions for training and generation using rectified flow.
"""

import torch
import math

from meteolibre_model.diffusion.utils import MEAN_CHANNEL, STD_CHANNEL

# -- Parameters --
CLIP_MIN = -4
COEF_NOISE = 2  


# --- Keep your existing helper functions ---
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def logsnr_schedule_cosine(t, logsnr_min=-15, logsnr_max=15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))

def logsnr_schedule_cosine_shifted(coef, t):
    logsnr_t = logsnr_schedule_cosine(t)
    return logsnr_t + 2 * math.log(coef)

def get_proper_noise(coef, t):
    t_shift = 1.0 - t
    logsnr_t = logsnr_schedule_cosine_shifted(coef, t_shift)
    beta_t = torch.sqrt(torch.sigmoid(logsnr_t))
    return 1.0 - beta_t, logsnr_t

def ds_dt(t):
    # This function now needs to handle a tensor of local times
    t_clamped = torch.clamp(t, min=1e-6, max=1 - 1e-6)
    t_shift = 1.0 - t_clamped
    logsnr_t = logsnr_schedule_cosine_shifted(COEF_NOISE, t_shift)
    sigmoid_logsnr_t = torch.sigmoid(logsnr_t)
    sqrt_sigmoid_logsnr_t = torch.sqrt(sigmoid_logsnr_t)
    d_sigmoid = sigmoid_logsnr_t * (1 - sigmoid_logsnr_t)
    t_min = math.atan(math.exp(-0.5 * 15))
    t_max = math.atan(math.exp(-0.5 * -15))
    tan_val = torch.tan(t_min + t_shift * (t_max - t_min))
    d_logsnr = -2 * (t_max - t_min) / tan_val * (1 / torch.cos(t_min + t_shift * (t_max - t_min)))**2
    d_beta_dt = (0.5 / sqrt_sigmoid_logsnr_t) * d_sigmoid * d_logsnr * (-1)
    return -d_beta_dt


# --- NEW: Rolling Diffusion Schedule Implementation ---
def get_rolling_local_times(global_t, W, schedule_type='lin', device='cuda'):
    """
    Computes the local diffusion times t_w for each frame in the window.
    Args:
        global_t (torch.Tensor): A scalar or batch of global times, shape (B,).
        W (int): The size of the sliding window (number of frames being predicted).
        schedule_type (str): 'lin' for linear rolling or 'init' for boundary condition.
        device (str): The device to create tensors on.
    Returns:
        torch.Tensor: Local times for each frame, shape (B, 1, W, 1, 1).
    """
    # Create a tensor for local indices w = [0, 1, ..., W-1]
    w_indices = torch.arange(W, device=device).float()

    # Reshape for broadcasting:
    # global_t: (B,) -> (B, 1)
    # w_indices: (W,) -> (1, W)
    global_t = global_t.view(-1, 1)

    if schedule_type == 'lin':
        # Equation (21) from the paper, adapted for a generic window
        # t_w = (w + t) / W
        local_t = (w_indices + global_t) / W
    elif schedule_type == 'init':
        # Equation (23) from the paper
        # t_w = clip(w/W + t)
        local_t = (w_indices / W) + global_t
    else:
        raise ValueError("schedule_type must be 'lin' or 'init'")

    # Clip values to be in [0, 1] as required
    local_t = torch.clamp(local_t, 0.0, 1.0)

    # Reshape for broadcasting with video tensor (B, C, W, H, W)
    # -> (B, W) -> (B, 1, W, 1, 1)
    return local_t.view(global_t.shape[0], 1, W, 1, 1)

def normalize(batch_data, device):
    """
    Normalize the batch data using precomputed mean and std.
    """
    batch_data = (
        batch_data
        - MEAN_CHANNEL.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    ) / STD_CHANNEL.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)

    # Clamp to prevent extreme values
    batch_data = batch_data.clamp(CLIP_MIN, 4)

    return batch_data

# Your existing get_x_t, but we'll call it with local times
def get_x_t(x0, x1, t_local):
    s_t, logsnr_t = get_proper_noise(COEF_NOISE, t_local)
    return s_t * x0 + (1 - s_t) * x1, logsnr_t

# --- MODIFIED: Trainer Step for Rolling Diffusion ---
def trainer_step_rolling(model, batch, device, parametrization="standard", beta=0.9):
    """
    Performs a single training step using the Rolling Diffusion strategy.

    Args:
        beta (float): The Bernoulli rate for oversampling the linear ('lin') schedule.
    """
    with torch.no_grad():
        batch_data = batch["patch_data"].permute(0, 2, 1, 3, 4)
        b, c, t, h, w = batch_data.shape
        batch_data = normalize(batch_data, device)

        x_context = batch_data[:, :, :4]  # Context frames
        x_target = batch_data[:, :, 4:] - batch_data[:, :, 3:4]  # Target residual
        mask_data = x_target != CLIP_MIN

        W = x_target.shape[2] # Window size is the number of target frames

        # Sample global timesteps
        global_t_batch = torch.rand(b, device=device)

        # --- Rolling Diffusion Core Logic ---
        # Decide which schedule to use based on Bernoulli sampling 
        use_lin_schedule = torch.bernoulli(torch.tensor([beta] * b)).bool().to(device)
        use_init_schedule = ~use_lin_schedule

        local_t_w = torch.zeros(b, 1, W, 1, 1, device=device)

        # Compute local times using the appropriate schedule for each item in the batch
        if torch.any(use_lin_schedule):
            local_t_w[use_lin_schedule] = get_rolling_local_times(
                global_t_batch[use_lin_schedule], W, 'lin', device
            )
        if torch.any(use_init_schedule):
            local_t_w[use_init_schedule] = get_rolling_local_times(
                global_t_batch[use_init_schedule], W, 'init', device
            )
        # --- End Core Logic ---

        # Generate noise (x1)
        x1 = torch.randn_like(x_target)

        # Get x_t using frame-dependent local times
        x_t, logsnr_t = get_x_t(x_target, x1, local_t_w)

        # Calculate target velocity
        if parametrization == "standard":
            # True velocity v = ds/dt * (x0 - x1)
            # ds_dt now gets a tensor of local times and returns a tensor of derivatives
            target = ds_dt(local_t_w) * (x_target - x1)
        elif parametrization == "endpoint":
            target = x_target
        else:
            raise ValueError(f"Unknown parametrization: {parametrization}")

    model_input = torch.cat([x_context, x_t], dim=2)
    context_info = batch["spatial_position"]
    # We pass the logsnr of the *first* frame in the window as context
    context_global = torch.cat([context_info, logsnr_t[:, 0, 0, 0, 0].view(-1, 1) / 10.0], dim=1)

    prediction = model(model_input.float(), context_global.float())[:, :, 4:, :, :]
    loss = torch.nn.functional.mse_loss(prediction[mask_data], target[mask_data])

    return loss

def solve_ode_for_window(model, x_context, x_t_start, context_info, steps, schedule_type, W, parametrization, device):
    """Helper function to solve the ODE for a single window."""
    x_t = x_t_start
    dt = 1.0 / steps

    for i in range(steps):
        t_val = 1.0 - i * dt
        global_t = torch.full((x_t.shape[0],), t_val, device=device)

        # Get local times for the current global t
        local_t_w = get_rolling_local_times(global_t, W, schedule_type, device)

        # Get interpolation coefficient and its derivative
        s_t, logsnr_t = get_proper_noise(COEF_NOISE, local_t_w)
        derivative = ds_dt(local_t_w)

        # Prepare context for the model
        context_global = torch.cat([context_info, logsnr_t[:, 0, 0, 0, 0].view(-1, 1) / 10.0], dim=1)

        # Predict velocity
        model_input = torch.cat([x_context, x_t], dim=2)
        pred_v = model(model_input.float(), context_global.float())[:, :, 4:, :, :]

        if parametrization == "endpoint":
            velocity = -derivative * (pred_v - x_t) / (s_t + 1e-6)
        else: # standard
            velocity = pred_v

        # Euler step
        x_t = x_t - velocity * dt
        x_t = x_t.clamp(-7, 7) # Clamp to prevent divergence

    return x_t


def rolling_generation_rollout(model, batch, num_frames_to_gen, window_size=2, steps_per_frame=100, device="cuda", parametrization="standard"):
    """
    Generates a long sequence of frames using the Rolling Diffusion rollout procedure.
    """
    model.eval()
    with torch.no_grad():
        model.to(device)

        # Prepare initial context
        x_context_full = batch["patch_data"].permute(0, 2, 1, 3, 4)[[0], :, :, :, :] # Use first batch item
        x_context_full = normalize(x_context_full, device)
        context_info = batch["spatial_position"].to(device)[[0], :]
        b, c, t, h, w = x_context_full.shape
        num_context_frames = 4

        # This will store the final generated sequence
        generated_frames = []

        # --- 1. Initialization Step (using 'init' schedule) ---
        # Generate the first `window_size` frames from pure noise [cite: 244]
        print("Step 1: Initializing first window...")
        initial_context = x_context_full[:, :, :num_context_frames]
        initial_noise = torch.randn(b, c, window_size, h, w, device=device)
        
        # This becomes the first fully-denoised window of frames (residuals)
        current_window_state = solve_ode_for_window(
            model, initial_context, initial_noise, context_info, steps_per_frame, 'init', window_size, parametrization, device
        )
        # Add the first generated frame to our final list
        generated_frames.append(current_window_state[:, :, [0]])

        # --- 2. Rollout Loop (using 'lin' schedule) ---
        print("Step 2: Starting rollout loop...")
        for k in range(1, num_frames_to_gen):
            print(f"  Generating frame {k+1}/{num_frames_to_gen}...")
            # The new context is the last N frames of the *full known sequence* so far
            # which is the original context + newly generated frames
            if k < num_context_frames:
                last_known_frame = x_context_full[:, :, [num_context_frames-1-k]]
            else:
                last_known_frame = generated_frames[k-num_context_frames]

            rollout_context = torch.cat([
                x_context_full[:, :, k:num_context_frames],
                *generated_frames
            ], dim=2)[:,:,-num_context_frames:]

            # The starting state for the ODE solve is the previous window's output,
            # shifted by one frame, with new noise appended[cite: 47].
            x_t_start = torch.cat([
                current_window_state[:, :, 1:],
                torch.randn(b, c, 1, h, w, device=device)
            ], dim=2)

            # Solve the ODE for the new sliding window
            current_window_state = solve_ode_for_window(
                model, rollout_context, x_t_start, context_info, steps_per_frame, 'lin', window_size, parametrization, device
            )
            
            # The fully generated frame is now the first one in the new window
            generated_frames.append(current_window_state[:, :, [0]])

    # --- 3. Finalize ---
    # Concatenate all generated frame residuals
    final_residuals = torch.cat(generated_frames, dim=2)
    
    # Add back the last context frame to convert residuals to actual frames
    # This requires reconstructing the sequence frame by frame
    final_sequence = []
    last_frame = x_context_full[:, :, [-1]]
    for i in range(final_residuals.shape[2]):
        next_frame = last_frame + final_residuals[:,:,[i]]
        final_sequence.append(next_frame)
        last_frame = next_frame

    model.train()
    return torch.cat(final_sequence, dim=2).cpu()

