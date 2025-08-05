"""
Code for https://arxiv.org/pdf/2412.03791v1
COORDINATE IN AND VALUE OUT: TRAINING FLOW
TRANSFORMERS IN AMBIENT SPACE

"""

import torch
import torch.nn as nn
import einops
from typing import Optional
from timm.models.vision_transformer import PatchEmbed
from dit_ml.dit import DiT


class CrossAttentionBlock(nn.Module):
    """
    A Cross-Attention block, a core component of a Transformer decoder.

    This block takes a query tensor `x` and a context tensor `context`. The query
    attends to the context to produce an updated representation of the query. This
    is followed by a standard feed-forward network (FFN).
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        """
        Initializes the CrossAttentionBlock.

        Args:
            embed_dim (int): The dimensionality of the input and output tensors.
            num_heads (int): The number of attention heads.
            mlp_ratio (float): Determines the hidden dimension of the FFN.
                               mlp_dim = embed_dim * mlp_ratio.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Layer normalization for inputs
        self.norm_x = nn.LayerNorm(embed_dim)
        self.norm_context = nn.LayerNorm(embed_dim)

        # Multi-head cross-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True  # Expects (B, seq_len, embed_dim)
        )

        # Layer normalization for the FFN input
        self.norm_ffn = nn.LayerNorm(embed_dim)

        # Feed-Forward Network (FFN)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CrossAttentionBlock.

        Args:
            x (torch.Tensor): The query tensor. Shape: (B, N_query, C).
            context (torch.Tensor): The context tensor (from the encoder).
                                    Shape: (B, N_context, C).

        Returns:
            torch.Tensor: The updated query tensor. Shape: (B, N_query, C).
        """
        # --- 1. Cross-Attention with pre-normalization and residual connection ---
        # The query `x` attends to the `context`.
        # Note: In nn.MultiheadAttention, if key/value are not provided, they default
        # to the query. Here we explicitly provide context for key and value.
        attn_output, _ = self.attention(
            query=self.norm_x(x),
            key=self.norm_context(context),
            value=self.norm_context(context)
        )
        # Residual connection
        x = x + attn_output

        # --- 2. Feed-Forward Network with pre-normalization and residual connection ---
        ffn_output = self.ffn(self.norm_ffn(x))
        # Residual connection
        x = x + ffn_output

        return x

class ASFTDecoder(nn.Module):
    """
    A simplified decoder for an Ambient Space Flow Transformer (ASFT).

    This decoder takes a set of query points (e.g., coordinate-value pairs) and
    a latent context vector `z_ft` from an encoder. It uses a series of
    CrossAttentionBlocks to produce the final output (e.g., velocity predictions).
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int = 1,
        output_dim: Optional[int] = None
    ):
        """
        Initializes the Decoder.

        Args:
            embed_dim (int): The dimensionality of the model.
            num_heads (int): The number of attention heads for each block.
            depth (int): The number of CrossAttentionBlocks to stack.
            output_dim (int, optional): The dimension of the final output. If None,
                                        it defaults to `embed_dim`.
        """
        super().__init__()
        
        # Stack of cross-attention blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads) for _ in range(depth)
        ])

        # Final projection head to map to the desired output dimension
        # (e.g., predicting a 3D velocity vector)
        self.proj_out = nn.Linear(embed_dim, output_dim or embed_dim)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Decoder.

        Args:
            queries (torch.Tensor): The input query points. Shape: (B, N_query, C_in).
                                    These are the points for which we want predictions.
            context (torch.Tensor): The context tensor from the encoder (z_ft).
                                    Shape: (B, N_context, C_in).

        Returns:
            torch.Tensor: The final predictions for each query point.
                          Shape: (B, N_query, C_out).
        """
        x = queries
        
        for block in self.blocks:
            # In each block, the queries are refined by attending to the same context
            x = block(x, context)
        
        # Project to the final output dimension
        output = self.proj_out(x)
        return output
    
    

class ASFTEncoder(nn.Module):
    """
    A simplified encoder for an Ambient Space Flow Transformer (ASFT).

    """
    def __init__(
        self,
        hidden_size: int = 384,
        num_heads: int = 8,
        depth: int = 12,
        condition_size: int = 3,
        in_channels: int = 5,
        nb_temporals: int = 6,
        output_dim: Optional[int] = None
    ):
        """
        Initializes the Decoder.

        Args:
            hidden_size (int): The dimensionality of the model.
            num_heads (int): The number of attention heads for each block.
            depth (int): The number of CrossAttentionBlocks to stack.
            output_dim (int, optional): The dimension of the final output. If None,
                                        it defaults to `hidden_size`.
        """
        super().__init__()
        
        self.nb_temporals = nb_temporals
        
        # projection to hidden size
        self.mlp = nn.Sequential(
            nn.Linear(condition_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        
        self.x_embedder = PatchEmbed(
            256,  # image size
            16,  # patch size
            in_channels,  # input channels
            hidden_size,  # hidden size
            bias=True,
        )

        self.encoder_model_core = DiT(
            num_patches=16 * 16 * nb_temporals,  # if 2d with flatten size
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            use_rope=False,
            rope_dimension=3,
            max_h=16,
            max_w=16,
            max_d=nb_temporals,
        )

    def forward(self, x_image, x_scalar):

        x_scalar = self.mlp(x_scalar)
        
        x = einops.rearrange(x_image, "b c n h w -> (b c) n h w")
        
        x = self.x_embedder(x)
        
        x = einops.rearrange(x, "(b n) nb_seq d -> b (n nb_seq) d", n=self.nb_temporals)
        x = self.encoder_model_core(x, x_scalar)
        
        return x