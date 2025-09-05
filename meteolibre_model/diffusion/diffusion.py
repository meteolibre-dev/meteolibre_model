"""
In this module we will use helper to create a proper diffusion setupcre
"""

import torch


def trainer_step(model, batch_data):
    """
    Performs a single training step for a rectified flow model.

    Args:
        model: The neural network model. It is expected to take a tensor of shape
               (BATCH, NB_CHANNEL, 6, H, W) and a timestep `t` as input, and
               return a velocity tensor of shape (BATCH, NB_CHANNEL, 2, H, W).
        batch_data: A tensor of shape (BATCH, 6, NB_CHANNEL, H, W), where the
                    first 4 time steps are the context and the last 2 are the target.

    Returns:
        The loss value for the training step.
    """
    # The model expects (BATCH, NB_CHANNEL, NB_TEMPORAL, H, W), so permute dimensions
    batch_data = batch_data.permute(0, 2, 1, 3, 4)

    x_context = batch_data[:, :, :4]  # Shape: (BATCH, NB_CHANNEL, 4, H, W)
    x_target = batch_data[:, :, 4:]  # Shape: (BATCH, NB_CHANNEL, 2, H, W)

    # 1. Generate the prior / noise (z)
    z = torch.randn_like(x_target)
    x_0 = z
    x_1 = x_target

    # 2. Generate a random t batch between 0 and 1
    t = torch.rand(batch_data.size(0), device=batch_data.device)
    # Reshape for broadcasting
    t_reshaped = t.view(-1, 1, 1, 1, 1)

    # 3. Create the interpolated data for the last two frames
    t_squared_reshaped = (t**2).view(-1, 1, 1, 1, 1)
    x_t = (1 - t_squared_reshaped) * x_0 + t_squared_reshaped * x_1

    # 4. Put the scalar values and the video values in the model and output the velocity
    # The model input is the concatenation of the context and the interpolated target.
    model_input = torch.cat(
        [x_context, x_t], dim=2
    )  # Shape: (BATCH, NB_CHANNEL, 6, H, W)

    # The model is expected to take the combined video and the timestep t.
    # NOTE: Your UNet_DCAE_3D model needs to be adapted to accept the timestep `t`
    # and to output a tensor of shape (BATCH, NB_CHANNEL, 2, H, W).
    predicted_velocity = model(model_input, t)

    # 5. Apply loss function
    target_velocity = 2 * t_reshaped * (x_1 - x_0)
    loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)

    return loss


def full_image_generation(model, x_context, steps=10, device="cuda", solver="euler"):
    """
    Generates full images by integrating from noise using a specified ODE solver.

    This function performs the full sampling process for a rectified flow model,
    to generate the target frames based on the context frames.

    Args:
        model: The neural network model. It is expected to take a tensor of shape
               (BATCH, NB_CHANNEL, 6, H, W) and a timestep `t` as input, and
               return a velocity tensor of shape (BATCH, NB_CHANNEL, 2, H, W).
        x_context: The context tensor of shape (BATCH, NB_CHANNEL, 4, H, W).
        steps (int): The number of integration steps for the solver.
        device: The device to perform generation on ('cuda' or 'cpu').
        solver (str): The ODE solver to use. Can be 'euler' or 'rk4'.

    Returns:
        The generated image tensor of shape (BATCH, NB_CHANNEL, 2, H, W).
    """
    model.eval()
    model.to(device)
    x_context = x_context.to(device)

    # 1. Generate the prior / noise (z) which is our starting point x_0
    batch_size, nb_channel, _, h, w = x_context.shape
    x_t = torch.randn(batch_size, nb_channel, 2, h, w, device=device)

    # 2. Perform integration from t=0 to t=1
    dt = 1.0 / steps

    def get_velocity(current_x, time_val):
        t = torch.full((batch_size,), time_val, device=device, dtype=torch.float32)
        model_input = torch.cat([x_context, current_x], dim=2)
        with torch.no_grad():
            return model(model_input, t)

    if solver == "euler":
        for i in range(steps):
            t_n = i * dt
            predicted_velocity = get_velocity(x_t, t_n)
            x_t = x_t + predicted_velocity * dt
    elif solver == "rk4":
        for i in range(steps):
            t_n = i * dt

            k1 = get_velocity(x_t, t_n)
            k2 = get_velocity(x_t + dt * k1 / 2, t_n + dt / 2)
            k3 = get_velocity(x_t + dt * k2 / 2, t_n + dt / 2)
            k4 = get_velocity(x_t + dt * k3, t_n + dt)

            x_t = x_t + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Set model back to training mode
    model.train()

    # The final x_t at t=1 is our generated sample
    return x_t.cpu()
