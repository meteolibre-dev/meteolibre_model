"""
In this module we will use helper to create a proper diffusion setupcre
"""

import torch


import numpy as np
import scipy.fft
from PIL import Image


# -- Parameters --
IMG_SIZE = 128
TIME_STEP = 0.2  # A single time t between 0.0 and 1.0
BLUR_SIGMA_MAX = 0.05 # From Table 3, a high-performing value [cite: 306]


# --- Step 1: Corrected schedule calculation ---

def get_blurring_diffusion_schedules(
    t: float, 
    height: int, 
    width: int, 
    sigma_b_max: float, 
    d_min: float = 0.001
):
    """
    Computes the combined blurring and noise schedules for a given time t.
    
    Args:
        t (float): Timestep between 0 and 1.
        height (int): Image height.
        width (int): Image width.
        sigma_b_max (float): Maximum blur sigma.
        d_min (float): Minimum damping factor for high frequencies.
        
    Returns:
        tuple: A tuple containing:
            - alpha_t_vec (np.ndarray): The frequency-dependent signal scaling vector.
            - sigma_t_scalar (float): The scalar noise standard deviation.
    """
    # 1. Standard Gaussian Noise Schedule (Cosine Schedule)
    # This defines the overall signal-to-noise ratio.
    # a_t = cos(t * pi/2) and sigma_t^2 = 1 - a_t^2
    a_t_scalar = np.cos(t * np.pi / 2.0)
    sigma_t_scalar = np.sin(t * np.pi / 2.0)

    # 2. Blurring Schedule (sin^2 schedule for blur amount)
    # sigma_B,t = sigma_B,max * sin(t*pi/2)^2
    sigma_b_t = sigma_b_max * (np.sin(t * np.pi / 2.0) ** 2)

    # 3. Calculate the frequency-dependent damping factor d_t
    if sigma_b_t > 0:
        # Dissipation time tau_t = sigma_B,t^2 / 2
        dissipation_time = (sigma_b_t ** 2) / 2.0
        
        # Calculate frequencies (lambda)
        # Note: The paper's appendix implies standard frequency calculation, not pi * freqs
        freqs_h = np.fft.fftfreq(height) * height
        freqs_w = np.fft.fftfreq(width) * width
        lambda_h = (2 * np.pi * freqs_h) ** 2
        lambda_w = (2 * np.pi * freqs_w) ** 2
        lambda_2d = lambda_h[:, np.newaxis] + lambda_w[np.newaxis, :]
        
        # d_t = (1 - d_min) * exp(-lambda * tau_t) + d_min
        d_t_vec = (1 - d_min) * np.exp(-lambda_2d * dissipation_time) + d_min
    else:
        # At t=0, there is no blurring
        d_t_vec = np.ones((height, width))

    # 4. Combine the schedules: alpha_t = a_t * d_t
    alpha_t_vec = a_t_scalar * d_t_vec
    
    return alpha_t_vec.reshape(1, 1, height, width), sigma_t_scalar

# --- Step 2: New function to add blurry noise at time t ---

def add_blurry_noise(image: np.ndarray, t: float, sigma_b_max: float) -> np.ndarray:
    """
    Applies the blurring diffusion forward process to an image for a single timestep t.
    """
    h, w = image.shape[2], image.shape[3]
    
    # Get the correct combined schedules for the given time t
    alpha_t, sigma_t = get_blurring_diffusion_schedules(t, h, w, sigma_b_max)

    # Transform image to frequency space using Discrete Cosine Transform (DCT)
    image_freq = scipy.fft.dctn(image, norm="ortho", axes=(-2, -1))
    
    # Generate random noise in pixel space
    noise = np.random.randn(*image.shape)

    # Apply the forward process: z_t = V(alpha_t * u_x) + sigma_t * epsilon
    blurred_signal_freq = alpha_t * image_freq
    blurred_signal = scipy.fft.idctn(blurred_signal_freq, norm="ortho", axes=(-2, -1))
    
    noisy_image = blurred_signal + sigma_t * noise
    return noisy_image
    
# --- Helper function to add standard isotropic noise (for comparison) ---


def trainer_step(model, batch_data):
    """
    Performs a single training step for a blurring diffusion model.

    Args:
        model: The neural network model. It is expected to take a tensor of shape
               (BATCH, NB_CHANNEL, 6, H, W) and a timestep `t` as input, and
               return the predicted noise of shape (BATCH, NB_CHANNEL, 2, H, W).
        batch_data: A tensor of shape (BATCH, 6, NB_CHANNEL, H, W), where the
                    first 4 time steps are the context and the last 2 are the target.

    Returns:
        The loss value for the training step.
    """
    # The model expects (BATCH, NB_CHANNEL, NB_TEMPORAL, H, W), so permute dimensions
    batch_data = batch_data.permute(0, 2, 1, 3, 4)

    x_context = batch_data[:, :, :4]  # Shape: (BATCH, NB_CHANNEL, 4, H, W)
    x_target = batch_data[:, :, 4:]   # This is x_0, shape: (BATCH, NB_CHANNEL, 2, H, W)

    # 1. Generate random timesteps for the batch
    t_batch = torch.rand(batch_data.size(0), device=batch_data.device)

    # 2. Generate noise
    noise = torch.randn_like(x_target)

    # 3. Create noisy target by applying blurring diffusion forward process
    x_t_batch = torch.zeros_like(x_target)
    
    # Loop over the batch to create x_t for each sample
    # This is done one-by-one because get_blurring_diffusion_schedules is not vectorized
    for i in range(x_target.size(0)):
        x_0_sample = x_target[i:i+1]
        t_sample = t_batch[i].item()
        noise_sample = noise[i:i+1]

        h, w = x_0_sample.shape[-2], x_0_sample.shape[-1]

        # Get schedules for the current timestep
        alpha_t_vec, sigma_t_scalar = get_blurring_diffusion_schedules(
            t_sample, h, w, BLUR_SIGMA_MAX
        )

        # Convert numpy schedules to torch tensors
        alpha_t_t = torch.from_numpy(alpha_t_vec).to(x_target.device, dtype=x_target.dtype)
        sigma_t_t = torch.tensor(sigma_t_scalar, device=x_target.device, dtype=x_target.dtype)

        # Move to numpy for DCT
        x_0_sample_np = x_0_sample.cpu().numpy()
        
        # Apply DCT
        x_0_freq = scipy.fft.dctn(x_0_sample_np, norm="ortho", axes=(-2, -1))

        # Apply blurring in frequency domain
        blurred_signal_freq = alpha_t_t.cpu().numpy() * x_0_freq

        # Apply IDCT
        blurred_signal_np = scipy.fft.idctn(blurred_signal_freq, norm="ortho", axes=(-2, -1))
        
        blurred_signal = torch.from_numpy(blurred_signal_np).to(x_target.device, dtype=x_target.dtype)

        # Add noise in pixel space to create x_t
        x_t_sample = blurred_signal + sigma_t_t * noise_sample
        x_t_batch[i] = x_t_sample

    # 4. Get model prediction
    # The model input is the concatenation of the context and the noisy target.
    model_input = torch.cat(
        [x_context, x_t_batch], dim=2
    )  # Shape: (BATCH, NB_CHANNEL, 6, H, W)

    # The model predicts the noise `epsilon` based on the noisy image `x_t` and timestep `t`.
    predicted_noise = model(model_input, t_batch)

    # 5. Apply loss function (MSE between actual and predicted noise)
    loss = torch.nn.functional.mse_loss(predicted_noise, noise)

    return loss


def full_image_generation(model, x_context, steps=1000, device="cuda"):
    """
    Generates full images using the DDPM ancestral sampler for Blurring Diffusion.

    This function implements Algorithm 1 from the paper[cite: 203]. It starts with pure
    noise and iteratively denoises it over a series of timesteps to generate the
    final target frames based on the context frames.

    Args:
        model: The neural network model that predicts noise `epsilon`.
        x_context: The context tensor of shape (BATCH, NB_CHANNEL, 4, H, W).
        steps (int): The number of denoising steps (T in the paper).
        device: The device to perform generation on ('cuda' or 'cpu').

    Returns:
        The generated image tensor of shape (BATCH, NB_CHANNEL, 2, H, W).
    """
    model.eval()
    model.to(device)
    x_context = x_context.to(device)
    
    batch_size, nb_channel, _, h, w = x_context.shape
    delta = 1e-8 # Small constant for numerical stability [cite: 472]

    # 1. Start with pure Gaussian noise (z_T) [cite: 204]
    z_t = torch.randn(batch_size, nb_channel, 2, h, w, device=device)

    # 2. Loop from T to 1 [cite: 205]
    for i in range(steps, 0, -1):
        t_val = i / steps
        s_val = (i - 1) / steps
        
        # Create a tensor for the current timestep
        t_batch = torch.full((batch_size,), t_val, device=device, dtype=torch.float32)

        # Get the model's noise prediction
        model_input = torch.cat([x_context, z_t], dim=2)
        with torch.no_grad():
            hat_eps_t = model(model_input, t_batch)
        
        # --- Prepare for denoising calculation in frequency space ---
        # Convert current state and predicted noise to numpy for DCT
        z_t_np = z_t.cpu().numpy()
        hat_eps_t_np = hat_eps_t.cpu().numpy()

        # Transform to frequency space u_t = V^T * z_t [cite: 205]
        u_t = scipy.fft.dctn(z_t_np, norm="ortho", axes=(-2, -1))
        hat_u_eps_t = scipy.fft.dctn(hat_eps_t_np, norm="ortho", axes=(-2, -1))

        # --- Calculate the parameters for the denoising distribution p(z_s | z_t) ---
        # Get schedules for current (t) and previous (s) timesteps
        alpha_t, sigma_t = get_blurring_diffusion_schedules(t_val, h, w, BLUR_SIGMA_MAX)
        alpha_s, sigma_s = get_blurring_diffusion_schedules(s_val, h, w, BLUR_SIGMA_MAX)
        
        # Calculate coefficients for the mean of the denoising distribution [cite: 216]
        alpha_ts = alpha_t / (alpha_s + delta)
        sigma2_ts = sigma_t**2 - alpha_ts**2 * sigma_s**2
        
        # Calculate denoising variance (posterior variance) [cite: 179]
        sigma2_denoise = 1.0 / np.maximum(
            (1.0 / np.maximum(sigma_s**2, delta)) + (alpha_ts**2 / np.maximum(sigma2_ts, delta)),
            delta
        )
        
        # Calculate denoising mean (hat_mu) [cite: 216]
        # This is a direct implementation of Equation 23
        coeff1 = sigma2_denoise * alpha_ts / np.maximum(sigma2_ts, delta)
        coeff2 = sigma2_denoise / (alpha_s * np.maximum(sigma_s**2, delta))
        
        term_in_eps = u_t - sigma_t * hat_u_eps_t
        hat_mu_t_s = coeff1 * u_t + coeff2 * term_in_eps

        # --- Sample z_s from the denoising distribution ---
        if i > 1:
            # Generate random noise for the sampling step
            noise_z = np.random.randn(*z_t.shape)
            u_noise = scipy.fft.dctn(noise_z, norm="ortho", axes=(-2,-1))
            
            # Sample u_s = mean + std * noise [cite: 207]
            u_s = hat_mu_t_s + np.sqrt(sigma2_denoise) * u_noise
        else:
            # The last step is deterministic
            u_s = hat_mu_t_s
            
        # Convert back to pixel space and update z_t for the next iteration
        z_s_np = scipy.fft.idctn(u_s, norm="ortho", axes=(-2, -1))
        z_t = torch.from_numpy(z_s_np).to(device, dtype=torch.float32)

    # Set model back to training mode
    model.train()

    # The final z_0 is our generated sample
    return z_t.cpu()