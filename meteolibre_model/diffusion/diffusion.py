"""
In this module we will use helper to create a proper diffusion setupcre
"""

import torch

MEAN_CHANNEL = torch.tensor(
    [
        0.93817195,
        1.19817447,
        0.12322853,
        0.93354392,
        0.26123831,
        36.65962429,
        28.19503977,
        62.73929266,
        80.87527095,
        71.9558302,
        2.5896961,
        11.48664509,
    ]
)

STD_CHANNEL = torch.tensor(
    [
        1.27078496,
        1.8195922,
        0.38173912,
        1.47627378,
        0.18876226,
        12.46953485,
        7.98968902,
        18.21728238,
        20.43442948,
        14.30025048,
        0.59147741,
        3.12146046,
    ]
)


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
    x_t = (1 - t_reshaped) * x_0 + t_reshaped * x_1

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
    target_velocity = x_1 - x_0
    loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)

    return loss
