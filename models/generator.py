"""
Simple U-Net generator for LDCT denoising.
Basic architecture without attention/dense/residual features.
This is the architecture that achieved +4.41 dB PSNR.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolutional block with two conv layers and leaky relu activations.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x


class EncoderBlock(nn.Module):
    """
    Encoder block: conv block followed by max pooling.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(EncoderBlock, self).__init__()

        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple:
        conv_out = self.conv_block(x)
        pooled = self.pool(conv_out)
        return conv_out, pooled


class DecoderBlock(nn.Module):
    """
    Decoder block: upsample, conv to reduce channels, concat skip, then conv block.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DecoderBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_reduce = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.conv_block = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.activation(self.conv_reduce(x))
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class UNetGenerator(nn.Module):
    """
    Simple U-Net generator for LDCT denoising.
    Basic architecture without attention/dense/residual features.

    Architecture:
        Encoder: 3 encoder blocks (64, 128, 256 channels)
        Bottleneck: conv block (512 channels)
        Decoder: 3 decoder blocks (256, 128, 64 channels)
        Output: 1x1 conv with tanh activation
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        **kwargs  # Accept but ignore extra arguments for compatibility
    ) -> None:
        super(UNetGenerator, self).__init__()

        # Encoder path
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        # Decoder path
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels)

        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=1, padding=0),
            nn.Tanh()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using kaiming normal for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through generator.

        Args:
            x: input tensor [batch, 1, 256, 256]

        Returns:
            output tensor [batch, 1, 256, 256] with values in [-1, 1]
        """
        # Encoder path with skip connections
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        # Output
        x = self.output(x)

        return x


if __name__ == '__main__':
    # Test generator
    print("=" * 50)
    print("Testing Simple UNet Generator")
    print("=" * 50)

    model = UNetGenerator()

    # Test with batch size 4
    x = torch.randn(4, 1, 256, 256)

    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("Generator test passed!")
