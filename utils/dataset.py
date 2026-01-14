"""
dataset module for loading paired ldct and ndct images for gan training.
supports both raw images (dicom, png, jpg) and preprocessed cached data.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# optional dicom support
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False


class LDCTDataset(Dataset):
    """
    pytorch dataset for loading paired low-dose ct and normal-dose ct images.

    supports two data sources:
    1. raw images: loads from directories with low_dose/ and normal_dose/ subdirectories
    2. cached/preprocessed: loads from .npz files created by preprocess_data.py

    attributes:
        root_dir: path to the root data directory
        mode: one of 'train', 'val', or 'test'
        use_cache: whether to load from preprocessed cache
        image_size: target size for resizing images (only used when use_cache=false)
    """

    # supported image formats for raw loading
    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.dcm'}

    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        image_size: int = 256,
        use_cache: bool = False
    ) -> None:
        """
        initialize the ldct dataset.

        args:
            root_dir: path to the root data directory
                      for raw: e.g., 'data/raw' (expects low_dose/ and normal_dose/ subdirs)
                      for cache: e.g., 'data/processed' (expects .npz files)
            mode: dataset split, one of 'train', 'val', or 'test'
            image_size: target image size for resizing (only used when use_cache=false)
            use_cache: if true, load preprocessed .npz files; if false, load raw images

        raises:
            assertionerror: if mode is invalid, directories don't exist,
                           or ldct/ndct counts don't match (raw mode only)
        """
        super().__init__()

        # validate mode
        valid_modes = {'train', 'val', 'test'}
        assert mode in valid_modes, f"mode must be one of {valid_modes}, got '{mode}'"

        self.root_dir = Path(root_dir)
        self.mode = mode
        self.image_size = image_size
        self.use_cache = use_cache

        if use_cache:
            self._init_cached()
        else:
            self._init_raw()

    def _init_cached(self) -> None:
        """initialize dataset for cached/preprocessed data loading."""
        self.cache_dir = self.root_dir / self.mode

        # verify cache directory exists
        assert self.cache_dir.exists(), f"cache directory not found: {self.cache_dir}"

        # collect .npz files
        self.cache_paths = sorted(self.cache_dir.glob('*.npz'))

        # verify we found cached files
        assert len(self.cache_paths) > 0, f"no .npz files found in {self.cache_dir}"

        # print summary
        print(f"Found {len(self.cache_paths)} cached pairs in {self.mode.upper()} set")

    def _init_raw(self) -> None:
        """initialize dataset for raw image loading."""
        # construct paths to low_dose and normal_dose directories
        self.ldct_dir = self.root_dir / self.mode / 'low_dose'
        self.ndct_dir = self.root_dir / self.mode / 'normal_dose'

        # verify directories exist
        assert self.ldct_dir.exists(), f"low_dose directory not found: {self.ldct_dir}"
        assert self.ndct_dir.exists(), f"normal_dose directory not found: {self.ndct_dir}"

        # collect and sort image paths
        self.ldct_paths = self._get_image_paths(self.ldct_dir)
        self.ndct_paths = self._get_image_paths(self.ndct_dir)

        # verify we found images
        assert len(self.ldct_paths) > 0, f"no images found in {self.ldct_dir}"
        assert len(self.ndct_paths) > 0, f"no images found in {self.ndct_dir}"

        # verify equal number of ldct and ndct images
        assert len(self.ldct_paths) == len(self.ndct_paths), (
            f"ldct and ndct image counts don't match: "
            f"{len(self.ldct_paths)} vs {len(self.ndct_paths)}"
        )

        # print summary
        print(f"Found {len(self.ldct_paths)} image pairs in {self.mode.upper()} set")

    def _get_image_paths(self, directory: Path) -> List[Path]:
        """
        get sorted list of image paths from a directory.

        args:
            directory: path to search for images

        returns:
            sorted list of image paths with supported extensions
        """
        paths = []
        for ext in self.SUPPORTED_EXTENSIONS:
            # case-insensitive matching
            paths.extend(directory.glob(f'*{ext}'))
            paths.extend(directory.glob(f'*{ext.upper()}'))

        # sort by filename to ensure ldct/ndct pairs align
        return sorted(paths, key=lambda p: p.stem)

    def _load_image(self, path: Path) -> np.ndarray:
        """
        load an image from file and convert to grayscale numpy array.

        args:
            path: path to the image file

        returns:
            grayscale image as numpy array with shape (h, w)
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

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        preprocess image: resize and normalize to [-1, 1].

        args:
            image: grayscale numpy array with values in [0, 255]

        returns:
            tensor with shape [1, h, w] and values in [-1, 1]
        """
        # convert to pil for resizing with bilinear interpolation
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize(
            (self.image_size, self.image_size),
            resample=Image.BILINEAR
        )

        # convert back to numpy
        image = np.array(pil_image, dtype=np.float32)

        # normalize from [0, 255] to [-1, 1]
        image = (image / 127.5) - 1.0

        # convert to tensor and add channel dimension [1, h, w]
        tensor = torch.from_numpy(image).unsqueeze(0)

        return tensor

    def __len__(self) -> int:
        """return the number of image pairs in the dataset."""
        if self.use_cache:
            return len(self.cache_paths)
        return len(self.ldct_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        get a paired ldct and ndct sample.

        args:
            idx: index of the sample to retrieve

        returns:
            tuple of (ldct_tensor, ndct_tensor), each with shape [1, h, w]
        """
        if self.use_cache:
            return self._getitem_cached(idx)
        return self._getitem_raw(idx)

    def _getitem_cached(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """load preprocessed data from cache."""
        data = np.load(self.cache_paths[idx])

        ldct = data['ldct']
        ndct = data['ndct']

        # convert to tensors and add channel dimension [1, h, w]
        ldct_tensor = torch.from_numpy(ldct).unsqueeze(0)
        ndct_tensor = torch.from_numpy(ndct).unsqueeze(0)

        return ldct_tensor, ndct_tensor

    def _getitem_raw(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """load and preprocess raw images."""
        # load images
        ldct_image = self._load_image(self.ldct_paths[idx])
        ndct_image = self._load_image(self.ndct_paths[idx])

        # preprocess to tensors
        ldct_tensor = self._preprocess(ldct_image)
        ndct_tensor = self._preprocess(ndct_image)

        return ldct_tensor, ndct_tensor

    def get_filenames(self, idx: int) -> Tuple[str, str]:
        """
        get the filenames for a given index (useful for debugging).

        args:
            idx: index of the sample

        returns:
            tuple of (ldct_filename, ndct_filename) for raw mode
            or (cache_filename, cache_filename) for cached mode
        """
        if self.use_cache:
            filename = self.cache_paths[idx].name
            return filename, filename
        return self.ldct_paths[idx].name, self.ndct_paths[idx].name


def create_dataloader(
    root_dir: str,
    mode: str,
    batch_size: int = 16,
    image_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_cache: bool = False
) -> torch.utils.data.DataLoader:
    """
    create a dataloader for the ldct dataset.

    args:
        root_dir: path to the root data directory
        mode: dataset split ('train', 'val', or 'test')
        batch_size: number of samples per batch
        image_size: target image size
        shuffle: whether to shuffle the data
        num_workers: number of worker processes for data loading
        pin_memory: whether to pin memory for faster gpu transfer
        use_cache: whether to use preprocessed cached data

    returns:
        pytorch dataloader instance
    """
    dataset = LDCTDataset(
        root_dir=root_dir,
        mode=mode,
        image_size=image_size,
        use_cache=use_cache
    )

    # adjust settings based on mode
    if mode in ['val', 'test']:
        shuffle = False

    # use 0 workers on windows to avoid multiprocessing issues
    import platform
    if platform.system() == 'Windows':
        num_workers = 0

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(mode == 'train')
    )

    return dataloader


if __name__ == '__main__':
    import time

    # test the dataset loading
    print("=" * 60)
    print("testing ldct dataset")
    print("=" * 60)

    # test raw loading
    print("\n--- testing raw data loading ---")
    try:
        dataset_raw = LDCTDataset('data/raw', mode='train', use_cache=False)
        print(f"dataset size: {len(dataset_raw)}")

        # benchmark raw loading
        start = time.time()
        ldct, ndct = dataset_raw[0]
        raw_time = time.time() - start

        print(f"ldct shape: {ldct.shape}, range [{ldct.min():.2f}, {ldct.max():.2f}]")
        print(f"ndct shape: {ndct.shape}, range [{ndct.min():.2f}, {ndct.max():.2f}]")
        print(f"load time (raw): {raw_time*1000:.2f} ms")

        # verify shapes
        assert ldct.shape == (1, 256, 256), f"unexpected ldct shape: {ldct.shape}"
        assert ndct.shape == (1, 256, 256), f"unexpected ndct shape: {ndct.shape}"

        print("raw loading test passed")

    except Exception as e:
        print(f"raw loading error: {e}")

    # test cached loading
    print("\n--- testing cached data loading ---")
    try:
        dataset_cached = LDCTDataset('data/processed', mode='train', use_cache=True)
        print(f"dataset size: {len(dataset_cached)}")

        # benchmark cached loading
        start = time.time()
        ldct, ndct = dataset_cached[0]
        cache_time = time.time() - start

        print(f"ldct shape: {ldct.shape}, range [{ldct.min():.2f}, {ldct.max():.2f}]")
        print(f"ndct shape: {ndct.shape}, range [{ndct.min():.2f}, {ndct.max():.2f}]")
        print(f"load time (cached): {cache_time*1000:.2f} ms")

        # verify shapes
        assert ldct.shape == (1, 256, 256), f"unexpected ldct shape: {ldct.shape}"
        assert ndct.shape == (1, 256, 256), f"unexpected ndct shape: {ndct.shape}"

        # compare speed
        if 'raw_time' in dir():
            speedup = raw_time / cache_time if cache_time > 0 else float('inf')
            print(f"speedup: {speedup:.2f}x faster")

        print("cached loading test passed")

    except AssertionError as e:
        print(f"cached data not found (run preprocess_data.py first): {e}")
    except Exception as e:
        print(f"cached loading error: {e}")

    print("\n" + "=" * 60)
    print("all tests completed")
    print("=" * 60)
