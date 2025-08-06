"""
meteolibre_model/dit/dit_core.py
"""

from mpmath.libmp import prec_to_dps

import torch
import torch.nn as nn
import einops

import lightning.pytorch as pl
from timm.models.vision_transformer import PatchEmbed

from dit_ml.dit import DiT


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x):
        # shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(self.norm_final(x))
        return x


class DiTCore(pl.LightningModule):
    """
    A modified version of the DiT model for use in the meteolibre_model.

    Will be use the preprocess the core latent image

    """

    def __init__(
        self, nb_temporals, hidden_size=384, depth=12, num_heads=8, patch_size=2, out_channels=16, in_channels=16, condition_size=3
    ):
        super().__init__()
        self.nb_temporals = nb_temporals
        self.out_channels = out_channels

        self.model_core = DiT(
            num_patches=16 * 16 * nb_temporals,  # if 2d with flatten size
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            use_rope=True,
            rope_dimension=3,
            max_h=16,
            max_w=16,
            max_d=nb_temporals,
        )

        self.x_embedder = PatchEmbed(
            32,  # image size
            patch_size,  # patch size
            in_channels,  # input channels
            hidden_size,  # hidden size
            bias=True,
        )

        # projection to hidden size
        self.mlp = nn.Sequential(
            nn.Linear(condition_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, scalar_input):
        """
        Forward pass of DiT.
            x: (N, nb_timestep, C, H, W) tensor of spatial inputs (images or latent representations of images)
            scalar: (N, condition_size) tensor of diffusion timesteps (or any other scalar input)
        """
        x_scalar = self.mlp(scalar_input)

        x = einops.rearrange(x, "b c n h w -> (b n) c h w")

        breakpoint()
        x = self.x_embedder(x)

        # resize temporals
        x = einops.rearrange(x, "(b n) nb_seq d -> b (n nb_seq) d", n=self.nb_temporals)

        x = self.model_core(x, x_scalar)

        x = einops.rearrange(x, "b (n nb_seq) d -> (b n) nb_seq d", n=self.nb_temporals)

        x = self.final_layer(x)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        x = einops.rearrange(x, "(b n) c h w -> b n c h w", n=self.nb_temporals)
        return x
