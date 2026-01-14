"""
training script for ldct gan.
implements wgan-gp training with combined loss.
"""

import argparse
from pathlib import Path
import time
from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from models.losses import WassersteinGPLoss, CombinedLoss
from utils.dataset import create_dataloader


class Trainer:
    """
    trainer class for ldct gan.
    handles training loop, checkpointing, and logging.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """
        initialize trainer.

        args:
            args: command line arguments
        """
        self.args = args

        # select best available device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # create output directories
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.log_dir = Path(args.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # initialize models
        print(f"initializing models on {self.device}...")
        self.generator = UNetGenerator(
            use_attention=args.use_attention,
            use_dense=args.use_dense,
            residual_learning=args.residual_learning
        ).to(self.device)
        self.discriminator = PatchGANDiscriminator().to(self.device)

        # print model info
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"  generator parameters: {g_params:,}")
        print(f"  discriminator parameters: {d_params:,}")

        # initialize optimizers
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=args.lr_g,
            betas=(args.beta1, args.beta2)
        )
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=args.lr_d,
            betas=(args.beta1, args.beta2)
        )

        # initialize learning rate schedulers
        if args.use_scheduler:
            self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_g, T_0=10, T_mult=2, eta_min=args.lr_g * 0.01
            )
            self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_d, T_0=10, T_mult=2, eta_min=args.lr_d * 0.01
            )
        else:
            self.scheduler_g = None
            self.scheduler_d = None

        # initialize losses
        self.criterion_g = CombinedLoss(
            lambda_adv=args.lambda_adv,
            lambda_perceptual=args.lambda_perceptual,
            lambda_l1=args.lambda_l1,
            lambda_ms_ssim=args.lambda_ms_ssim,
            lambda_edge=args.lambda_edge,
            use_wgan=True,
            use_perceptual=args.use_perceptual,
            use_ms_ssim=args.use_ms_ssim,
            use_edge=args.use_edge
        ).to(self.device)

        self.criterion_d = WassersteinGPLoss(lambda_gp=args.lambda_gp)

        # initialize dataloaders
        print("initializing dataloaders...")
        self.train_loader = create_dataloader(
            root_dir=args.data_dir,
            mode='train',
            batch_size=args.batch_size,
            image_size=args.image_size,
            shuffle=True,
            num_workers=args.num_workers,
            use_cache=args.use_cache
        )

        self.val_loader = create_dataloader(
            root_dir=args.data_dir,
            mode='val',
            batch_size=args.batch_size,
            image_size=args.image_size,
            shuffle=False,
            num_workers=args.num_workers,
            use_cache=args.use_cache
        )

        print(f"  train samples: {len(self.train_loader.dataset)}")
        print(f"  val samples: {len(self.val_loader.dataset)}")

        # initialize tensorboard
        self.writer = SummaryWriter(log_dir=args.log_dir)

        # training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_epoch(self) -> Dict[str, float]:
        """
        train for one epoch.

        returns:
            dictionary of average metrics for the epoch
        """
        self.generator.train()
        self.discriminator.train()

        metrics = {}
        pbar = tqdm(self.train_loader, desc=f"epoch {self.epoch}")

        for ldct, ndct in pbar:
            ldct = ldct.to(self.device)
            ndct = ndct.to(self.device)

            # train discriminator
            for _ in range(self.args.n_critic):
                self.optimizer_d.zero_grad()

                # generate fake images
                with torch.no_grad():
                    fake = self.generator(ldct)

                # discriminator loss
                loss_d, metrics_d = self.criterion_d.discriminator_loss(
                    self.discriminator, ndct, fake
                )

                loss_d.backward()
                self.optimizer_d.step()

            # train generator
            self.optimizer_g.zero_grad()

            # generate fake images
            fake = self.generator(ldct)

            # generator loss
            loss_g, metrics_g = self.criterion_g.generator_loss(
                fake, ndct, self.discriminator
            )

            loss_g.backward()
            self.optimizer_g.step()

            # update learning rate schedulers
            if self.scheduler_g is not None:
                self.scheduler_g.step(self.epoch + self.global_step / len(self.train_loader))
            if self.scheduler_d is not None:
                self.scheduler_d.step(self.epoch + self.global_step / len(self.train_loader))

            # accumulate metrics
            for k, v in {**metrics_d, **metrics_g}.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)

            # update progress bar
            pbar.set_postfix({
                'g_loss': f"{metrics_g['g_total_loss']:.4f}",
                'd_loss': f"{metrics_d['d_loss']:.4f}",
                'w_dist': f"{metrics_d.get('wasserstein_dist', 0):.4f}"
            })

            # log to tensorboard
            if self.global_step % self.args.log_interval == 0:
                for k, v in {**metrics_d, **metrics_g}.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)

            self.global_step += 1

        # average metrics
        avg_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
        return avg_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        validate on validation set.

        returns:
            dictionary of average validation metrics
        """
        self.generator.eval()
        self.discriminator.eval()

        metrics = {}
        pbar = tqdm(self.val_loader, desc="validating")

        for ldct, ndct in pbar:
            ldct = ldct.to(self.device)
            ndct = ndct.to(self.device)

            # generate fake images
            fake = self.generator(ldct)

            # compute generator loss (no discriminator backward needed)
            _, metrics_g = self.criterion_g.generator_loss(
                fake, ndct, self.discriminator
            )

            # simple discriminator scores (no gradient penalty for validation)
            d_real = self.discriminator(ndct)
            d_fake = self.discriminator(fake)

            # compute simple wasserstein loss (no GP for validation)
            d_loss = d_fake.mean() - d_real.mean()

            metrics_d = {
                'd_loss': d_loss.item(),
                'd_real': d_real.mean().item(),
                'd_fake': d_fake.mean().item(),
                'wasserstein_dist': (d_real.mean() - d_fake.mean()).item()
            }

            # accumulate metrics
            for k, v in {**metrics_d, **metrics_g}.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)

        # average metrics
        avg_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}

        # log to tensorboard
        for k, v in avg_metrics.items():
            self.writer.add_scalar(f'val/{k}', v, self.epoch)

        # log sample images
        if self.epoch % self.args.save_image_interval == 0:
            self.writer.add_images('val/ldct', (ldct[:4] + 1) / 2, self.epoch)
            self.writer.add_images('val/fake', (fake[:4] + 1) / 2, self.epoch)
            self.writer.add_images('val/ndct', (ndct[:4] + 1) / 2, self.epoch)

        return avg_metrics

    def save_checkpoint(self, filename: str = 'checkpoint.pth') -> None:
        """
        save checkpoint.

        args:
            filename: checkpoint filename
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'best_val_loss': self.best_val_loss,
            'args': self.args
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"checkpoint saved: {path}")

    def load_checkpoint(self, filename: str = 'checkpoint.pth') -> None:
        """
        load checkpoint.

        args:
            filename: checkpoint filename
        """
        path = self.checkpoint_dir / filename

        if not path.exists():
            print(f"checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"checkpoint loaded: {path}")
        print(f"  resuming from epoch {self.epoch}")

    def train(self) -> None:
        """main training loop."""
        print(f"\nstarting training for {self.args.num_epochs} epochs")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(self.epoch, self.args.num_epochs):
            self.epoch = epoch

            # train
            train_metrics = self.train_epoch()

            # validate
            if (epoch + 1) % self.args.val_interval == 0:
                val_metrics = self.validate()

                # print metrics
                print(f"\nepoch {epoch + 1}/{self.args.num_epochs}")
                print(f"  train - g_loss: {train_metrics['g_total_loss']:.4f}, "
                      f"d_loss: {train_metrics['d_loss']:.4f}")
                print(f"  val   - g_loss: {val_metrics['g_total_loss']:.4f}, "
                      f"d_loss: {val_metrics['d_loss']:.4f}")

                # save best model
                if val_metrics['g_total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['g_total_loss']
                    self.save_checkpoint('best.pth')
                    print(f"  new best model saved (val_loss: {self.best_val_loss:.4f})")

            # save checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
                self.save_checkpoint('latest.pth')

        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"training completed in {elapsed_time / 3600:.2f} hours")
        print(f"best validation loss: {self.best_val_loss:.4f}")

        self.writer.close()


def main():
    """main function."""
    parser = argparse.ArgumentParser(description='train ldct gan')

    # data
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='path to data directory')
    parser.add_argument('--use-cache', action='store_true', default=True,
                        help='use preprocessed cache')
    parser.add_argument('--image-size', type=int, default=256,
                        help='image size')

    # model architecture
    parser.add_argument('--use-attention', action='store_true', default=False,
                        help='use attention blocks in generator')
    parser.add_argument('--use-dense', action='store_true', default=False,
                        help='use dense connections in bottleneck')
    parser.add_argument('--residual-learning', action='store_true', default=False,
                        help='use residual learning (predict noise)')

    # loss weights
    parser.add_argument('--lambda-adv', type=float, default=1.0,
                        help='adversarial loss weight')
    parser.add_argument('--lambda-perceptual', type=float, default=10.0,
                        help='perceptual loss weight')
    parser.add_argument('--lambda-l1', type=float, default=100.0,
                        help='l1 loss weight')
    parser.add_argument('--lambda-ms-ssim', type=float, default=84.0,
                        help='multi-scale ssim loss weight')
    parser.add_argument('--lambda-edge', type=float, default=50.0,
                        help='edge-aware loss weight')
    parser.add_argument('--lambda-gp', type=float, default=10.0,
                        help='gradient penalty weight')

    # loss toggles
    parser.add_argument('--use-perceptual', action='store_true', default=False,
                        help='use perceptual loss')
    parser.add_argument('--use-ms-ssim', action='store_true', default=False,
                        help='use multi-scale ssim loss')
    parser.add_argument('--use-edge', action='store_true', default=False,
                        help='use edge-aware loss')

    # training
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--lr-g', type=float, default=0.0002,
                        help='generator learning rate')
    parser.add_argument('--lr-d', type=float, default=0.0002,
                        help='discriminator learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='adam beta2')
    parser.add_argument('--n-critic', type=int, default=5,
                        help='number of discriminator updates per generator update')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--use-scheduler', action='store_true', default=False,
                        help='use cosine annealing learning rate scheduler')

    # checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='runs',
                        help='tensorboard log directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume from checkpoint')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='checkpoint save interval (epochs)')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='validation interval (epochs)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='tensorboard logging interval (steps)')
    parser.add_argument('--save-image-interval', type=int, default=5,
                        help='save sample images interval (epochs)')

    args = parser.parse_args()

    # print configuration
    print("=" * 60)
    print("ldct gan training")
    print("=" * 60)
    print("configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("=" * 60)

    # initialize trainer
    trainer = Trainer(args)

    # resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # start training
    trainer.train()


if __name__ == '__main__':
    main()
