"""
Script to visualize the noise schedule for the score-based diffusion model.
"""

import torch
import matplotlib.pyplot as plt

# Constants from score_based.py
BETA_MIN = 0.1
BETA_MAX = 20.0


def beta_t(t):
    """Compute beta(t) = BETA_MIN + t * (BETA_MAX - BETA_MIN)"""
    return BETA_MIN + t * (BETA_MAX - BETA_MIN)


def sigma_t_sq(t):
    """Compute sigma_t^2 = integral_0^t beta(s) ds"""
    return BETA_MIN * t + (BETA_MAX - BETA_MIN) * t**2 / 2


def main():
    # Generate t values from 0 to 1
    t = torch.linspace(0, 1, 100)

    # Compute beta(t) and sigma_t^2(t)
    beta = beta_t(t)
    sigma_sq = sigma_t_sq(t)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot beta(t)
    ax1.plot(t.numpy(), beta.numpy(), label=r"$\beta(t)$", color="blue")
    ax1.set_title(r"$\beta(t) = 0.1 + t \cdot (20.0 - 0.1)$")
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$\beta(t)$")
    ax1.grid(True)
    ax1.legend()

    # Plot sigma_t^2(t)
    ax2.plot(t.numpy(), sigma_sq.numpy(), label=r"$\sigma_t^2(t)$", color="red")
    ax2.set_title(r"$\sigma_t^2(t) = \int_0^t \beta(s) \, ds$")
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"$\sigma_t^2(t)$")
    ax2.grid(True)
    ax2.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig("/workspace/meteolibre_model/noise_schedule.png", dpi=300)
    print(
        "Noise schedule visualization saved to /workspace/meteolibre_model/noise_schedule.png"
    )

    # Also print some values
    print(
        f"At t=0: beta={beta_t(torch.tensor(0.0)):.4f}, sigma_sq={sigma_t_sq(torch.tensor(0.0)):.4f}"
    )
    print(
        f"At t=0.5: beta={beta_t(torch.tensor(0.5)):.4f}, sigma_sq={sigma_t_sq(torch.tensor(0.5)):.4f}"
    )
    print(
        f"At t=1: beta={beta_t(torch.tensor(1.0)):.4f}, sigma_sq={sigma_t_sq(torch.tensor(1.0)):.4f}"
    )


if __name__ == "__main__":
    main()
