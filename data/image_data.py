import os
from typing import Dict, List, Tuple

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
    # Also remove the alpha channel
    img = img[crop_x_l:crop_x_r, crop_y_l:crop_y_r, :3]

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
        continual: bool,
    ):
        """Initialize an image fitting dataset.
        Args:
            side_length: Side length of image (i.e. number of pixels on each side).
            data_split: Data split of the BSDS dataset (i.e. 'train' or 'test').
            id: ID of the single image to use.
            region_simulator: Region simulator object to simulate regions that will be
                trained sequentially under continual learning.
            continual: If True, __len__ and __getitem__ pick from the current region.
                Else, they pick from the full available dataset.
        """
        super().__init__()
        spatial_dim = 2
        rgb_dim = 3

        # Load full image and its coordinates
        self.full_pixels = get_img_tensor_square(path, side_length)
        self.full_coords = get_mgrid(side_length, spatial_dim)

        # Split the image and coordinates into regions
        (
            self.coords_regions,
            self.pixels_regions,
        ) = region_simulator.simulate_regions(self.full_coords, self.full_pixels)
        self.continual = continual

        # Change input regions to dictionary format
        for i in range(len(self.coords_regions)):
            self.coords_regions[i] = {"coords": self.coords_regions[i]}

        # Flatten spatial dimension to obtain a batch of coordinates for training
        self.full_pixels = self.full_pixels.reshape(-1, rgb_dim)
        self.full_coords = self.full_coords.reshape(-1, spatial_dim)

        # Initialize current region to be the first region
        self.set_cur_region(0)

    def set_cur_region(self, region: int) -> None:
        self._cur_region = region
        self.pixels = self.pixels_regions[self.cur_region]
        self.coords = self.coords_regions[self.cur_region]

    def __len__(self):
        """Number of pixels in either the current or full grid."""
        if self.continual:
            return self.output.shape[0]
        else:
            return self.full_output.shape[0]

    def __getitem__(self, idx) -> Tuple[Dict[str, Tensor], Tensor]:
        """Return pixel at a coordinate either on the current or full grid.

        Returns: 2-tuple for input and output. Input is kwargs dictionary to feed into
        the network. Output is a tensor to match the network output.
            Input: 'coords': (2,) Coordinate in image.
            Output: (3,) Pixel value of this ray.
        """
        if self.continual:
            return {"coords": self.input["coords"][idx]}, self.output[idx]
        else:
            return {"coords": self.full_input["coords"][idx]}, self.full_output[idx]

    @property
    def cur_region(self) -> int:
        return self._cur_region

    @property
    def num_regions(self) -> int:
        return len(self.pixels_regions)

    @property
    def input(self) -> Tensor:
        return self.coords

    @property
    def output(self) -> Tensor:
        return self.pixels

    @property
    def full_input(self) -> Dict[str, Tensor]:
        return {"coords": self.full_coords}

    @property
    def full_output(self) -> Tensor:
        return self.full_pixels

    @property
    def input_regions(self) -> List:
        return self.coords_regions

    @property
    def output_regions(self) -> List:
        return self.pixels_regions
