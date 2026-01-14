"""
preprocessing script for ldct dataset.
converts raw images (dicom/png/jpg) to preprocessed numpy arrays for faster training.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

# optional dicom support
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False


# supported image formats
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.dcm'}


def get_image_paths(directory: Path) -> List[Path]:
    """
    get sorted list of image paths from a directory.

    args:
        directory: path to search for images

    returns:
        sorted list of image paths with supported extensions
    """
    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        # case-insensitive matching
        paths.extend(directory.glob(f'*{ext}'))
        paths.extend(directory.glob(f'*{ext.upper()}'))

    # sort by filename to ensure ldct/ndct pairs align
    return sorted(paths, key=lambda p: p.stem)


def load_image(path: Path) -> np.ndarray:
    """
    load an image from file and convert to grayscale numpy array.

    args:
        path: path to the image file

    returns:
        grayscale image as numpy array with shape (h, w) and uint8 dtype
    """
    ext = path.suffix.lower()

    if ext == '.dcm':
        # load dicom file
        if not DICOM_AVAILABLE:
            raise ImportError(
                "pydicom is required to load .dcm files. "
                "install with: pip install pydicom"
            )

        dcm = pydicom.dcmread(str(path))
        image = dcm.pixel_array.astype(np.float32)

        # apply rescale slope and intercept if available
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            image = image * dcm.RescaleSlope + dcm.RescaleIntercept

        # apply windowing for ct images (typical soft tissue window)
        # window center: 40 hu, window width: 400 hu
        window_center = 40
        window_width = 400
        window_min = window_center - window_width // 2
        window_max = window_center + window_width // 2

        image = np.clip(image, window_min, window_max)

        # normalize to 0-255 range
        image = ((image - window_min) / (window_max - window_min) * 255).astype(np.uint8)

    else:
        # load png/jpg/jpeg with pillow
        image = Image.open(path)

        # convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')

        image = np.array(image)

    return image


def preprocess_image(image: np.ndarray, image_size: int = 256) -> np.ndarray:
    """
    preprocess image: resize and normalize to [-1, 1].

    args:
        image: grayscale numpy array with values in [0, 255]
        image_size: target size for resizing

    returns:
        preprocessed array with shape (h, w) and values in [-1, 1], float32 dtype
    """
    # convert to pil for resizing with bilinear interpolation
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize(
        (image_size, image_size),
        resample=Image.BILINEAR
    )

    # convert back to numpy
    image = np.array(pil_image, dtype=np.float32)

    # normalize from [0, 255] to [-1, 1]
    image = (image / 127.5) - 1.0

    return image


def process_split(
    raw_dir: Path,
    processed_dir: Path,
    split: str,
    image_size: int = 256
) -> Tuple[int, int]:
    """
    process all images in a data split.

    args:
        raw_dir: path to raw data directory (e.g., data/raw)
        processed_dir: path to processed data directory (e.g., data/processed)
        split: one of 'train', 'val', 'test'
        image_size: target image size

    returns:
        tuple of (num_pairs_processed, total_bytes_saved)
    """
    # construct paths
    ldct_dir = raw_dir / split / 'low_dose'
    ndct_dir = raw_dir / split / 'normal_dose'
    output_dir = processed_dir / split

    # verify source directories exist
    if not ldct_dir.exists():
        print(f"  skipping {split}: low_dose directory not found")
        return 0, 0
    if not ndct_dir.exists():
        print(f"  skipping {split}: normal_dose directory not found")
        return 0, 0

    # get image paths
    ldct_paths = get_image_paths(ldct_dir)
    ndct_paths = get_image_paths(ndct_dir)

    if len(ldct_paths) == 0:
        print(f"  skipping {split}: no images found in low_dose")
        return 0, 0
    if len(ndct_paths) == 0:
        print(f"  skipping {split}: no images found in normal_dose")
        return 0, 0

    # verify counts match
    if len(ldct_paths) != len(ndct_paths):
        print(f"  error in {split}: ldct ({len(ldct_paths)}) and ndct ({len(ndct_paths)}) counts don't match")
        return 0, 0

    # create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # process each pair
    total_bytes = 0
    num_pairs = len(ldct_paths)

    print(f"  processing {num_pairs} pairs...")
    for idx in tqdm(range(num_pairs), desc=f"  {split}", unit="pair"):
        # load and preprocess images
        ldct_raw = load_image(ldct_paths[idx])
        ndct_raw = load_image(ndct_paths[idx])

        ldct_processed = preprocess_image(ldct_raw, image_size)
        ndct_processed = preprocess_image(ndct_raw, image_size)

        # save as npz
        output_path = output_dir / f"pair_{idx:04d}.npz"
        np.savez_compressed(
            output_path,
            ldct=ldct_processed,
            ndct=ndct_processed
        )

        total_bytes += output_path.stat().st_size

    return num_pairs, total_bytes


def format_bytes(size_bytes: int) -> str:
    """format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def main():
    """main preprocessing function."""
    parser = argparse.ArgumentParser(
        description='preprocess ldct dataset for faster training'
    )
    parser.add_argument(
        '--raw-dir',
        type=str,
        default='data/raw',
        help='path to raw data directory (default: data/raw)'
    )
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed',
        help='path to processed data directory (default: data/processed)'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=256,
        help='target image size (default: 256)'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'val', 'test'],
        help='splits to process (default: train val test)'
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)

    print("=" * 60)
    print("ldct data preprocessing")
    print("=" * 60)
    print(f"raw directory: {raw_dir}")
    print(f"output directory: {processed_dir}")
    print(f"image size: {args.image_size}x{args.image_size}")
    print(f"splits: {args.splits}")
    print()

    # verify raw directory exists
    if not raw_dir.exists():
        print(f"error: raw directory not found: {raw_dir}")
        sys.exit(1)

    # process each split
    total_pairs = 0
    total_bytes = 0

    for split in args.splits:
        print(f"processing {split} split:")
        pairs, bytes_saved = process_split(
            raw_dir,
            processed_dir,
            split,
            args.image_size
        )
        total_pairs += pairs
        total_bytes += bytes_saved
        if pairs > 0:
            print(f"  saved {pairs} pairs ({format_bytes(bytes_saved)})")
        print()

    # print summary
    print("=" * 60)
    print("preprocessing complete")
    print("=" * 60)
    print(f"total pairs processed: {total_pairs}")
    print(f"total disk space: {format_bytes(total_bytes)}")
    print(f"output location: {processed_dir}")


if __name__ == '__main__':
    main()
