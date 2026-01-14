"""
Evaluation script for LDCT-GAN model.
Evaluates the trained generator on the test set and computes metrics.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.generator import UNetGenerator
from utils.dataset import LDCTDataset


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 2.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1: First image tensor
        img2: Second image tensor
        max_val: Maximum possible pixel value (2.0 for range [-1, 1])

    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    Simplified implementation for grayscale images.

    Args:
        img1: First image tensor [1, H, W]
        img2: Second image tensor [1, H, W]
        window_size: Size of the Gaussian window

    Returns:
        SSIM value between -1 and 1 (1 is perfect similarity)
    """
    C1 = (0.01 * 2) ** 2  # Constants to stabilize division
    C2 = (0.03 * 2) ** 2

    # Remove channel dimension for calculation
    img1 = img1.squeeze(0)
    img2 = img2.squeeze(0)

    # Calculate means
    mu1 = img1.mean()
    mu2 = img2.mean()

    # Calculate variances and covariance
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    # Calculate SSIM
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim.item()


def save_comparison_images(ldct: torch.Tensor, ndct: torch.Tensor, fake: torch.Tensor,
                           save_path: Path, num_samples: int = 8):
    """
    Save comparison images showing LDCT, Generated, and NDCT side by side.

    Args:
        ldct: Low-dose CT images [B, 1, H, W]
        ndct: Normal-dose CT images [B, 1, H, W]
        fake: Generated images [B, 1, H, W]
        save_path: Path to save the comparison image
        num_samples: Number of samples to display
    """
    num_samples = min(num_samples, ldct.size(0))

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Convert from [-1, 1] to [0, 1] for display
        ldct_img = (ldct[i, 0].cpu().numpy() + 1) / 2
        fake_img = (fake[i, 0].cpu().numpy() + 1) / 2
        ndct_img = (ndct[i, 0].cpu().numpy() + 1) / 2

        # Display images
        axes[i, 0].imshow(ldct_img, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('Low-Dose CT (Input)')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(fake_img, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Generated (Output)')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(ndct_img, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title('Normal-Dose CT (Target)')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison images to {save_path}")


def evaluate(args):
    """
    Main evaluation function.
    """
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    generator = UNetGenerator(in_channels=1, out_channels=1).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        generator.load_state_dict(checkpoint)

    generator.eval()

    # Load test dataset
    print(f"Loading test dataset from {args.data_dir}")
    test_dataset = LDCTDataset(
        root_dir=args.data_dir,
        mode='test',
        image_size=args.image_size,
        use_cache=args.use_cache
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"Test set size: {len(test_dataset)}")

    # Evaluation metrics
    psnr_ldct_list = []  # PSNR between LDCT and NDCT (before)
    psnr_fake_list = []  # PSNR between Generated and NDCT (after)
    ssim_ldct_list = []  # SSIM between LDCT and NDCT (before)
    ssim_fake_list = []  # SSIM between Generated and NDCT (after)

    # Evaluate
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for batch_idx, (ldct, ndct) in enumerate(tqdm(test_loader)):
            ldct = ldct.to(device)
            ndct = ndct.to(device)

            # Generate images
            fake = generator(ldct)

            # Calculate metrics for each image in batch
            for i in range(ldct.size(0)):
                # PSNR
                psnr_ldct = calculate_psnr(ldct[i], ndct[i])
                psnr_fake = calculate_psnr(fake[i], ndct[i])
                psnr_ldct_list.append(psnr_ldct)
                psnr_fake_list.append(psnr_fake)

                # SSIM
                ssim_ldct = calculate_ssim(ldct[i].cpu(), ndct[i].cpu())
                ssim_fake = calculate_ssim(fake[i].cpu(), ndct[i].cpu())
                ssim_ldct_list.append(ssim_ldct)
                ssim_fake_list.append(ssim_fake)

            # Save comparison images for first batch
            if batch_idx == 0:
                save_comparison_images(
                    ldct, ndct, fake,
                    output_dir / 'comparison.png',
                    num_samples=min(8, ldct.size(0))
                )

    # Calculate average metrics
    avg_psnr_ldct = np.mean(psnr_ldct_list)
    avg_psnr_fake = np.mean(psnr_fake_list)
    avg_ssim_ldct = np.mean(ssim_ldct_list)
    avg_ssim_fake = np.mean(ssim_fake_list)

    psnr_improvement = avg_psnr_fake - avg_psnr_ldct
    ssim_improvement = avg_ssim_fake - avg_ssim_ldct

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nTest set size: {len(test_dataset)} images")
    print(f"\n{'Metric':<20} {'Before (LDCT)':<15} {'After (Generated)':<18} {'Improvement':<15}")
    print("-"*60)
    print(f"{'PSNR (dB)':<20} {avg_psnr_ldct:>14.4f} {avg_psnr_fake:>17.4f} {psnr_improvement:>14.4f}")
    print(f"{'SSIM':<20} {avg_ssim_ldct:>14.4f} {avg_ssim_fake:>17.4f} {ssim_improvement:>14.4f}")
    print("="*60)

    # Interpret results
    print("\nINTERPRETATION:")
    if psnr_improvement > 0:
        print(f"PSNR improved by {psnr_improvement:.4f} dB ")
    else:
        print(f"PSNR decreased by {abs(psnr_improvement):.4f} dB - Image quality is WORSE")

    if ssim_improvement > 0:
        print(f"SSIM improved by {ssim_improvement:.4f}")
    else:
        print(f"SSIM decreased by {abs(ssim_improvement):.4f}")

    # Save results to file
    results_file = output_dir / 'results.txt'
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("LDCT-GAN EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {args.checkpoint}\n")
        f.write(f"Test set size: {len(test_dataset)} images\n\n")
        f.write(f"{'Metric':<20} {'Before (LDCT)':<15} {'After (Generated)':<18} {'Improvement':<15}\n")
        f.write("-"*60 + "\n")
        f.write(f"{'PSNR (dB)':<20} {avg_psnr_ldct:>14.4f} {avg_psnr_fake:>17.4f} {psnr_improvement:>14.4f}\n")
        f.write(f"{'SSIM':<20} {avg_ssim_ldct:>14.4f} {avg_ssim_fake:>17.4f} {ssim_improvement:>14.4f}\n")
        f.write("="*60 + "\n")

    print(f"\nResults saved to {results_file}")
    print(f"Comparison images saved to {output_dir / 'comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate LDCT-GAN model')

    # Data parameters
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='path to processed data directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth',
                        help='path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='directory to save evaluation results')
    parser.add_argument('--use-cache', action='store_true', default=True,
                        help='use cached preprocessed data')
    parser.add_argument('--image-size', type=int, default=256,
                        help='image size for evaluation')

    # Evaluation parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of data loading workers')

    args = parser.parse_args()

    evaluate(args)


if __name__ == '__main__':
    main()
