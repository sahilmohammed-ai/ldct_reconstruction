"""
patchgan discriminator for ldct gan.
outputs a grid of real/fake predictions for 16x16 patches.
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    patchgan discriminator for ldct gan.

    architecture:
        4 conv blocks with stride 2 for downsampling
        output layer produces 16x16 patch predictions

    input: [batch, 1, 256, 256]
    output: [batch, 1, 16, 16] (raw logits, no activation)
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
        """
        initialize patchgan discriminator.

        args:
            in_channels: input channels (default 1 for grayscale)
            base_channels: base number of channels (default 64)
        """
        super(PatchGANDiscriminator, self).__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels

        # conv block 1: no batchnorm
        # [b, 1, 256, 256] -> [b, 64, 128, 128]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # conv block 2: with batchnorm
        # [b, 64, 128, 128] -> [b, 128, 64, 64]
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # conv block 3: with batchnorm
        # [b, 128, 64, 64] -> [b, 256, 32, 32]
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # conv block 4: with batchnorm
        # [b, 256, 32, 32] -> [b, 512, 16, 16]
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # output layer: no batchnorm, no activation (raw logits)
        # [b, 512, 16, 16] -> [b, 1, 16, 16] with k=3,s=1,p=1
        self.output = nn.Conv2d(base_channels * 8, 1, kernel_size=3, stride=1, padding=1)

        # initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """initialize weights using kaiming normal for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass through discriminator.

        args:
            x: input tensor [batch, 1, 256, 256]

        returns:
            output tensor [batch, 1, 16, 16] (patch predictions as raw logits)
        """
        x = self.conv1(x)   # [b, 64, 128, 128]
        x = self.conv2(x)   # [b, 128, 64, 64]
        x = self.conv3(x)   # [b, 256, 32, 32]
        x = self.conv4(x)   # [b, 512, 16, 16]
        x = self.output(x)  # [b, 1, 16, 16]

        return x


if __name__ == '__main__':
    # test discriminator
    print("=" * 50)
    print("testing patchgan discriminator")
    print("=" * 50)

    model = PatchGANDiscriminator()

    # test with batch size 4
    x = torch.randn(4, 1, 256, 256)

    print(f"input shape: {x.shape}")
    output = model(x)
    print(f"output shape: {output.shape}")
    print(f"output range: [{output.min():.3f}, {output.max():.3f}]")

    # verify output shape
    assert output.shape == (4, 1, 16, 16), f"unexpected output shape: {output.shape}"

    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total parameters: {total_params:,}")
    print(f"trainable parameters: {trainable_params:,}")

    # test with batch size 1
    x_single = torch.randn(1, 1, 256, 256)
    output_single = model(x_single)
    assert output_single.shape == (1, 1, 16, 16), "batch size 1 test failed"
    print("batch size 1 test passed")

    print("discriminator test passed")
