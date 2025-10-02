import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# ==============================================================================
# == 3D U-Net Blocks
# ==============================================================================


class DownsampleBlock3D(nn.Module):
    """
    3D Downsample Block.
    Downsamples spatial dimensions (H, W) by 2x, doubles channels, and keeps depth (D) intact.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if out_channels != 2 * in_channels:
            print(
                f"Warning: out_channels ({out_channels}) is not double the in_channels ({in_channels})."
            )

        # Main path: 3D strided convolution, only striding on H and W
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the non-parametric shortcut for 3D tensors.
        This is a custom implementation of pixel_unshuffle for 5D tensors (N, C, D, H, W).
        """
        N, C, D, H, W = x.shape
        # Reshape to create grid and then permute to bring grid elements to channel dimension
        x_reshaped = x.view(N, C, D, H // 2, 2, W // 2, 2)
        x_permuted = x_reshaped.permute(0, 1, 4, 5, 2, 3, 6).contiguous()
        s2c = x_permuted.view(N, C * 4, D, H // 2, W // 2)

        # Channel Averaging
        c_new = s2c.shape[1]
        group1 = s2c[:, : c_new // 2, :, :, :]
        group2 = s2c[:, c_new // 2 :, :, :, :]
        return (group1 + group2) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the 3D downsampling block."""
        return self.conv(x) + self._shortcut(x)


class UpsampleBlock3D(nn.Module):
    """
    3D Upsample Block.
    Upsamples spatial dimensions (H, W) by 2x, halves channels, and keeps depth (D) intact.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if out_channels != in_channels // 2:
            print(
                f"Warning: out_channels ({out_channels}) is not half the in_channels ({in_channels})."
            )

        # Main path: 3D transposed convolution, only upsampling H and W
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=0,
        )

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the non-parametric shortcut for 3D tensors.
        This is a custom implementation of pixel_shuffle for 5D tensors (N, C, D, H, W).
        """
        N, C, D, H, W = x.shape
        # The inverse of the unshuffle operation
        c2s = x.view(N, C // 4, 2, 2, D, H, W)
        c2s_permuted = c2s.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        c2s = c2s_permuted.view(N, C // 4, D, H * 2, W * 2)

        # Channel Duplicating
        return torch.cat([c2s, c2s], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the 3D upsampling block."""
        return self.conv_transpose(x) + self._shortcut(x)


# ==============================================================================
# == 3D U-Net Components
# ==============================================================================


class ResNetBlock3D(nn.Module):
    """
    A 3D ResNet block with two (1, 3, 3) convolutions and a skip connection.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Main path of the ResNet block
        self.main_path = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
        )

        # Shortcut connection to match dimensions if necessary
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )

        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_relu(self.main_path(x) + self.shortcut(x))


# ==============================================================================
# == Full 3D U-Net Architecture
# ==============================================================================
class UNet3D(nn.Module):
    """
    A 3D U-Net architecture that only performs spatial down/up-sampling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256],
    ):
        super().__init__()
        self.features = features  # Your correct addition!
        self.encoder_convs = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # --- Encoder (Downsampling Path) ---
        current_channels = in_channels
        for feature in features:
            self.encoder_convs.append(ResNetBlock3D(current_channels, feature))
            self.downs.append(DownsampleBlock3D(feature, feature * 2))
            current_channels = (
                feature * 2
            )  # CORRECTED: Update channels for the *next* block

        # --- Bottleneck ---
        # CORRECTED: The bottleneck input channels must match the last downsampler's output
        bottleneck_channels = features[-1] * 2
        self.bottleneck = ResNetBlock3D(bottleneck_channels, bottleneck_channels)

        # --- Decoder (Upsampling Path) ---
        for feature in reversed(features):
            self.ups.append(UpsampleBlock3D(feature * 2, feature))
            self.decoder_convs.append(ResNetBlock3D(feature * 2, feature))

        # --- Final Output Layer ---
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # --- Encoder Path ---
        # This forward pass logic was already correct and works with the fixed __init__
        for i in range(len(self.features)):
            x = self.encoder_convs[i](x)
            skip_connections.append(x)
            x = self.downs[i](x)

        # --- Bottleneck ---
        x = self.bottleneck(x)

        # --- Decoder Path ---
        skip_connections = skip_connections[::-1]
        for i in range(len(self.ups)):
            x = self.ups[i](x)
            skip_connection = skip_connections[i]

            # In case of rounding issues with odd dimensions
            if x.shape != skip_connection.shape:
                # Interpolate only H and W, keeping D the same.
                x = F.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder_convs[i](concat_skip)

        return self.final_conv(x)


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing Full 3D U-Net with ResNet Blocks ---")

    # Define model parameters
    IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH = 16, 128, 128
    IN_CHANNELS = 1
    OUT_CHANNELS = 1  # For binary segmentation
    BATCH_SIZE = 2

    # Create a random input tensor (N, C, D, H, W)
    input_tensor = torch.randn(
        BATCH_SIZE, IN_CHANNELS, IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH
    )
    print(f"Input shape: {input_tensor.shape}")

    # Initialize the model
    # Note: I adjusted the features list slightly for a more typical U-Net progression
    model = UNet3D(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, features=[32, 64, 128, 256]
    )

    # Perform a forward pass
    output_tensor = model(input_tensor)

    print(f"Output shape: {output_tensor.shape}")

    # Verify the output shape is as expected
    expected_shape = (BATCH_SIZE, OUT_CHANNELS, IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH)
    assert output_tensor.shape == expected_shape, (
        f"Shape mismatch! Expected {expected_shape}, got {output_tensor.shape}"
    )

    print("✅ 3D U-Net model shape test PASSED.")

    # Print model parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
