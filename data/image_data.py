import os

import hydra
import skimage.io as io
import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.region_simulators import RegionSimulator


def get_mgrid(side_length: int, dim: int = 2) -> Tensor:
    """Generate a flattened grid of (x, y, ...) coordinates in a range of -1 to 1.
    Args:
        side_length: Side length of the grid (i.e. number of points).
        dim: Dimension of the grid.
    Returns:
        (side_length, side_length, dim) A tensor of `dim`-dimensional coordinates.
    """
    axes = tuple(dim * [torch.linspace(-1, 1, steps=side_length)])
    mgrid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    return mgrid


def get_img_tensor_square(img_path: str, side_length: int) -> Tensor:
    """Return an image as a tensor, cropped into a square.
    Args:
        img_path: Path of image.
        side_length: Side length of the square to crop the image into.
    """
    try:
        # Internal use by hydra
        path = os.path.join(hydra.utils.get_original_cwd(), img_path)
    except:
        # External use by model inspector
        path = img_path

    img = torch.tensor(io.imread(path) / 255).float()

    # Resize image
    crop_x_l = (img.shape[0] - side_length) // 2
    crop_x_r = side_length + crop_x_l
    crop_y_l = (img.shape[1] - side_length) // 2
    crop_y_r = side_length + crop_y_l
    img = img[crop_x_l:crop_x_r, crop_y_l:crop_y_r, :]

    return img


class ImageFitting(Dataset):
    """Dataset that represents a single image for network fitting. One (x, y) data point
    represents one coordinate and its corresponding pixel value in the image.
    """

    def __init__(
        self,
        side_length: int,
        path: str,
        region_simulator: RegionSimulator,
    ):
        """Initialize an image fitting dataset.
        Args:
            side_length: Side length of image (i.e. number of pixels on each side).
            data_split: Data split of the BSDS dataset (i.e. 'train' or 'test').
            id: ID of the single image to use.
            region_simulator: Region simulator object to simulate regions that will be
                trained sequentially under continual learning.
        """
        super().__init__()
        spatial_dim = 2
        rgb_dim = 3

        # Load full image and its coordinates
        self.full_pixels = get_img_tensor_square(path, side_length)
        self.full_coords = get_mgrid(side_length, spatial_dim)

        # Split the image and coordinates into regions
        (
            self.pixels_regions,
            self.coords_regions,
        ) = region_simulator.simulate_regions(self.full_pixels, self.full_coords)

        # Flatten spatial dimension to obtain a batch of coordinates for training
        self.full_pixels = self.full_pixels.reshape(-1, rgb_dim)
        self.full_coords = self.full_coords.reshape(-1, spatial_dim)

        # Initialize current region to be the first region
        self.set_cur_region(0)

    def set_cur_region(self, region: int) -> None:
        self.cur_region = region
        self.pixels = self.pixels_regions[self.cur_region]
        self.coords = self.coords_regions[self.cur_region]

    @property
    def num_regions(self) -> int:
        return len(self.pixels_regions)

    def __len__(self):
        """Number of pixels in *full* grid."""
        return self.full_pixels.shape[0]

    def __getitem__(self, idx):
        """Return pixel at any coordinate on the *full* grid."""
        return self.full_coords[idx], self.full_pixels[idx]
