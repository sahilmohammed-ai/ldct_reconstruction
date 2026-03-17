"""
loss functions for ldct gan training.
simple version with just wgan-gp and l1 loss.
this is the loss configuration that achieved +4.41 db psnr.
"""

import torch
import torch.nn as nn
from typing import Tuple


class WassersteinGPLoss(nn.Module):
    """
    wasserstein gan loss with gradient penalty.
    more stable training than standard gan loss.
    """

    def __init__(self, lambda_gp: float = 10.0) -> None:
        """
        initialize wgan-gp loss.

        args:
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
        compute gradient penalty for wgan-gp.

        args:
            discriminator: discriminator network
            real: real images
            fake: generated images

        returns:
            gradient penalty value
        """
        batch_size = real.size(0)
        device = real.device

        # random interpolation coefficient
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)

        # interpolate between real and fake
        interpolates = alpha * real + (1 - alpha) * fake
        interpolates.requires_grad_(True)

        # discriminator output for interpolates
        d_interpolates = discriminator(interpolates)

        # compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # flatten gradients
        gradients = gradients.view(batch_size, -1)

        # compute gradient norm
        gradient_norm = gradients.norm(2, dim=1)

        # gradient penalty
        penalty = ((gradient_norm - 1) ** 2).mean()

        return penalty

    def discriminator_loss(
        self,
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        compute discriminator loss.

        args:
            discriminator: discriminator network
            real: real images
            fake: generated images (detached)

        returns:
            tuple of (loss, metrics_dict)
        """
        # discriminator outputs
        d_real = discriminator(real)
        d_fake = discriminator(fake.detach())

        # wasserstein loss
        loss_real = -d_real.mean()
        loss_fake = d_fake.mean()
        loss_w = loss_real + loss_fake

        # gradient penalty
        gp = self.gradient_penalty(discriminator, real, fake)
        loss_gp = self.lambda_gp * gp

        # total discriminator loss
        loss_d = loss_w + loss_gp

        # metrics
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
        compute generator adversarial loss.

        args:
            discriminator: discriminator network
            fake: generated images

        returns:
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
    combined loss for generator training.
    simple version: adversarial + l1 reconstruction loss only.
    """

    def __init__(
        self,
        lambda_adv: float = 1.0,
        lambda_l1: float = 100.0,
        lambda_gp: float = 10.0,
        **kwargs  # Accept but ignore extra arguments for compatibility
    ) -> None:
        """
        initialize combined loss.

        args:
            lambda_adv: weight for adversarial loss
            lambda_l1: weight for l1 reconstruction loss
            lambda_gp: weight for gradient penalty
        """
        super(CombinedLoss, self).__init__()

        self.lambda_adv = lambda_adv
        self.lambda_l1 = lambda_l1

        # adversarial loss
        self.adv_loss = WassersteinGPLoss(lambda_gp=lambda_gp)

        # pixel-wise loss
        self.l1_loss = nn.L1Loss()

    def generator_loss(
        self,
        fake: torch.Tensor,
        real: torch.Tensor,
        discriminator: nn.Module
    ) -> Tuple[torch.Tensor, dict]:
        """
        compute total generator loss.

        args:
            fake: generated images
            real: target real images
            discriminator: discriminator network

        returns:
            tuple of (total_loss, metrics_dict)
        """
        metrics = {}

        # adversarial loss
        loss_adv, adv_metrics = self.adv_loss.generator_loss(discriminator, fake)
        metrics.update(adv_metrics)

        # l1 loss
        loss_l1 = self.l1_loss(fake, real)
        metrics['l1_loss'] = loss_l1.item()

        # total loss
        total_loss = self.lambda_adv * loss_adv + self.lambda_l1 * loss_l1

        metrics['g_total_loss'] = total_loss.item()

        return total_loss, metrics


if __name__ == '__main__':
    # test loss functions
    print("=" * 50)
    print("Testing Simple Loss Functions")
    print("=" * 50)

    # create dummy data
    batch_size = 2
    fake = torch.randn(batch_size, 1, 256, 256)
    real = torch.randn(batch_size, 1, 256, 256)

    # test l1 loss
    print("\nTesting L1 loss:")
    l1_loss = nn.L1Loss()
    loss = l1_loss(fake, real)
    print(f"  L1 loss: {loss.item():.4f}")

    # test wgan-gp loss
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

    # test combined loss
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
