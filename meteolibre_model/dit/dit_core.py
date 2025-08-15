"""
meteolibre_model/dit/dit_core.py
"""

from mpmath.libmp import prec_to_dps

import torch
import torch.nn as nn
import einops

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


class DiTCore(nn.Module):
    """
    A modified version of the DiT model for use in the meteolibre_model.

    Will be use the preprocess the core latent image

    """

    def __init__(
        self,
        nb_temporals,
        hidden_size=384,
        depth=12,
        num_heads=8,
        patch_size=2,
        out_channels=16,
        in_channels=16,
        condition_size=3,
        image_size=16,
    ):
        super().__init__()
        self.nb_temporals = nb_temporals
        self.out_channels = out_channels
        self.image_size = image_size

        self.model_core = DiT(
            num_patches=image_size
            * image_size
            * nb_temporals,  # if 2d with flatten size
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            use_rope=True,
            rope_dimension=3,
            max_h=image_size,
            max_w=image_size,
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

        # proj init
        self.prof_init = nn.Linear(in_channels, hidden_size, bias=True)

        # proj end
        self.prof_end = nn.Linear(hidden_size, out_channels, bias=True)

        # time embedding (only for the time sequence)
        self.time_embedding = nn.Parameter(
            torch.randn(in_channels, nb_temporals)
            .unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

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

        x = x + self.time_embedding

        x = einops.rearrange(x, "b c n h w -> b (n h w) c")

        x = self.prof_init(x)
        # x = self.x_embedder(x)

        # resize temporals
        # x = einops.rearrange(x, "(b n) nb_seq d -> b (n nb_seq) d", n=self.nb_temporals)

        x = self.model_core(x, x_scalar)
        x = self.prof_end(x)

        # x = einops.rearrange(x, "b (n nb_seq) d -> (b n) nb_seq d", n=self.nb_temporals)

        # x = self.final_layer(x)  # (N, T, patch_size ** 2 * out_channels)
        # x = self.unpatchify(x)  # (N, out_channels, H, W)

        x = einops.rearrange(
            x,
            "b (n h w) c -> b c n h w",
            n=self.nb_temporals,
            h=self.image_size,
            w=self.image_size,
        )

        return x
