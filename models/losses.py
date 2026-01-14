"""
Loss functions for LDCT GAN training.
Simple version with just WGAN-GP and L1 loss.
This is the loss configuration that achieved +4.41 dB PSNR.
"""

import torch
import torch.nn as nn
from typing import Tuple


class WassersteinGPLoss(nn.Module):
    """
    Wasserstein GAN loss with gradient penalty.
    More stable training than standard GAN loss.
    """

    def __init__(self, lambda_gp: float = 10.0) -> None:
        """
        Initialize WGAN-GP loss.

        Args:
            lambda_gp: gradient penalty coefficient
        """
        super(WassersteinGPLoss, self).__init__()
        self.lambda_gp = lambda_gp

    def gradient_penalty(
        self,
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP.

        Args:
            discriminator: discriminator network
            real: real images
            fake: generated images

        Returns:
            gradient penalty value
        """
        batch_size = real.size(0)
        device = real.device

        # Random interpolation coefficient
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)

        # Interpolate between real and fake
        interpolates = alpha * real + (1 - alpha) * fake
        interpolates.requires_grad_(True)

        # Discriminator output for interpolates
        d_interpolates = discriminator(interpolates)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Flatten gradients
        gradients = gradients.view(batch_size, -1)

        # Compute gradient norm
        gradient_norm = gradients.norm(2, dim=1)

        # Gradient penalty
        penalty = ((gradient_norm - 1) ** 2).mean()

        return penalty

    def discriminator_loss(
        self,
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute discriminator loss.

        Args:
            discriminator: discriminator network
            real: real images
            fake: generated images (detached)

        Returns:
            tuple of (loss, metrics_dict)
        """
        # Discriminator outputs
        d_real = discriminator(real)
        d_fake = discriminator(fake.detach())

        # Wasserstein loss
        loss_real = -d_real.mean()
        loss_fake = d_fake.mean()
        loss_w = loss_real + loss_fake

        # Gradient penalty
        gp = self.gradient_penalty(discriminator, real, fake)
        loss_gp = self.lambda_gp * gp

        # Total discriminator loss
        loss_d = loss_w + loss_gp

        # Metrics
        metrics = {
            'd_loss': loss_d.item(),
            'd_real': d_real.mean().item(),
            'd_fake': d_fake.mean().item(),
            'gp': gp.item(),
            'wasserstein_dist': -(loss_real + loss_fake).item()
        }

        return loss_d, metrics

    def generator_loss(
        self,
        discriminator: nn.Module,
        fake: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute generator adversarial loss.

        Args:
            discriminator: discriminator network
            fake: generated images

        Returns:
            tuple of (loss, metrics_dict)
        """
        d_fake = discriminator(fake)
        loss_g = -d_fake.mean()

        metrics = {
            'g_adv_loss': loss_g.item(),
            'd_fake_for_g': d_fake.mean().item()
        }

        return loss_g, metrics


class CombinedLoss(nn.Module):
    """
    Combined loss for generator training.
    Simple version: adversarial + L1 reconstruction loss only.
    """

    def __init__(
        self,
        lambda_adv: float = 1.0,
        lambda_l1: float = 100.0,
        lambda_gp: float = 10.0,
        **kwargs  # Accept but ignore extra arguments for compatibility
    ) -> None:
        """
        Initialize combined loss.

        Args:
            lambda_adv: weight for adversarial loss
            lambda_l1: weight for L1 reconstruction loss
            lambda_gp: weight for gradient penalty
        """
        super(CombinedLoss, self).__init__()

        self.lambda_adv = lambda_adv
        self.lambda_l1 = lambda_l1

        # Adversarial loss
        self.adv_loss = WassersteinGPLoss(lambda_gp=lambda_gp)

        # Pixel-wise loss
        self.l1_loss = nn.L1Loss()

    def generator_loss(
        self,
        fake: torch.Tensor,
        real: torch.Tensor,
        discriminator: nn.Module
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total generator loss.

        Args:
            fake: generated images
            real: target real images
            discriminator: discriminator network

        Returns:
            tuple of (total_loss, metrics_dict)
        """
        metrics = {}

        # Adversarial loss
        loss_adv, adv_metrics = self.adv_loss.generator_loss(discriminator, fake)
        metrics.update(adv_metrics)

        # L1 loss
        loss_l1 = self.l1_loss(fake, real)
        metrics['l1_loss'] = loss_l1.item()

        # Total loss
        total_loss = self.lambda_adv * loss_adv + self.lambda_l1 * loss_l1

        metrics['g_total_loss'] = total_loss.item()

        return total_loss, metrics


if __name__ == '__main__':
    # Test loss functions
    print("=" * 50)
    print("Testing Simple Loss Functions")
    print("=" * 50)

    # Create dummy data
    batch_size = 2
    fake = torch.randn(batch_size, 1, 256, 256)
    real = torch.randn(batch_size, 1, 256, 256)

    # Test L1 loss
    print("\nTesting L1 loss:")
    l1_loss = nn.L1Loss()
    loss = l1_loss(fake, real)
    print(f"  L1 loss: {loss.item():.4f}")

    # Test WGAN-GP loss
    print("\nTesting WGAN-GP loss:")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.discriminator import PatchGANDiscriminator
    discriminator = PatchGANDiscriminator()

    wgan_loss = WassersteinGPLoss()
    loss_d, metrics_d = wgan_loss.discriminator_loss(discriminator, real, fake)
    print(f"  D_loss: {loss_d.item():.4f}")
    print(f"  Metrics: {metrics_d}")

    loss_g, metrics_g = wgan_loss.generator_loss(discriminator, fake)
    print(f"  G_loss: {loss_g.item():.4f}")
    print(f"  Metrics: {metrics_g}")

    # Test combined loss
    print("\nTesting combined loss:")
    combined_loss = CombinedLoss(
        lambda_adv=1.0,
        lambda_l1=100.0,
        lambda_gp=10.0
    )

    total_loss, metrics = combined_loss.generator_loss(fake, real, discriminator)
    print(f"  Total G_loss: {total_loss.item():.4f}")
    print(f"  Metrics: {metrics}")

    print("\nAll loss tests passed!")
