import unittest
import torch
from meteolibre_model.models.unet3d import (
    DownsampleBlock3D,
    UpsampleBlock3D,
    ResNetBlock3D,
    UNet3D,
)


class Test3DUNet(unittest.TestCase):
    def test_downsample_block(self):
        in_channels = 16
        out_channels = 32
        block = DownsampleBlock3D(in_channels, out_channels)
        input_tensor = torch.randn(2, in_channels, 16, 64, 64)
        output_tensor = block(input_tensor)
        self.assertEqual(output_tensor.shape, (2, out_channels, 16, 32, 32))

    def test_upsample_block(self):
        in_channels = 64
        out_channels = 32
        block = UpsampleBlock3D(in_channels, out_channels)
        input_tensor = torch.randn(2, in_channels, 16, 32, 32)
        output_tensor = block(input_tensor)
        self.assertEqual(output_tensor.shape, (2, out_channels, 16, 64, 64))

    def test_resnet_block(self):
        in_channels = 32
        out_channels = 64
        block = ResNetBlock3D(in_channels, out_channels)
        input_tensor = torch.randn(2, in_channels, 16, 64, 64)
        output_tensor = block(input_tensor)
        self.assertEqual(output_tensor.shape, (2, out_channels, 16, 64, 64))

    def test_unet3d(self):
        in_channels = 1
        out_channels = 1
        features = [32, 64]
        model = UNet3D(in_channels, out_channels, features)
        input_tensor = torch.randn(1, in_channels, 16, 128, 128)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, (1, out_channels, 16, 128, 128))


if __name__ == "__main__":
    unittest.main()
