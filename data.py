import os
from typing import List, Tuple, Union

import hydra
import skimage.io as io
import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    # Internal use by hydra (for instantiation)
    bsds300_dir = f"{hydra.utils.get_original_cwd()}/BSDS300/images"
except:
    # External use by model inspector
    bsds300_dir = "BSDS300/images"


def get_mgrid(side_length: int, dim: int = 2) -> Tensor:
    """Generate a flattened grid of (x, y, ...) coordinates in a range of -1 to 1.
    Args:
        side_length: Side length of the grid (i.e. number of points).
        dim: Dimension of the grid.
    Returns:
        (side_length, side_length, dim) A tensor of `dim`-dimensional coordinates.
    """
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=side_length)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    return mgrid


def get_bsds_tensor_square(side_length: int, split: str, id: str) -> Tensor:
    """Return a BSDS image as a tensor, cropped into a square.
    Args:
        side_length: Side length of the square to crop the image into.
        split: Split of BSDS image.
        id: ID of the BSDS image.
    """
    dir = os.path.join(bsds300_dir, split, f"{id}.jpg")
    img = torch.tensor(io.imread(dir) / 255).float()

    # Resize image
    crop_x_l = (img.shape[0] - side_length) // 2
    crop_x_r = side_length + crop_x_l
    crop_y_l = (img.shape[1] - side_length) // 2
    crop_y_r = side_length + crop_y_l
    img = img[crop_x_l:crop_x_r, crop_y_l:crop_y_r, :]

    return img


class RegionSimulator:
    def simulate_regions(
        self, full_pixels: Tensor, full_coords: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Simulate regions.
        Args:
            full_pixels: (side_length, side_length, value_dim) Full set of ground truth
                pixels.
            full_coords: (side_length, side_length, spatial_dim) Full set of ground
                truth coordinates corresponding to `full_pixels`.

        Returns:
            pixels_regions: List of pixel regions.
            coords_regions: Corresponding coordinate regions.
        """
        raise NotImplementedError


class RegularRegionSimulator(RegionSimulator):
    """Regular region simulator that produces square patches of the same size, covering
    the entire image evenly (except when `overlap` is nonzero, in which case regions
    will have different sizes). Regions contain ground truth coordinates.
    """

    def __init__(self, divide_side_n: int, overlap: int, permute: bool) -> None:
        """
        Args:
            divide_side_n: Evenly divide each side into n segments. This results in n^2
                regions to fit sequentially if using continual learning.
            overlap: Number of pixels to grow for each region (in the direction of
                increasing indices) to result in overlapping regions. If negative,
                regions shrink to not share edges at all.
            permute: Whether to randomly permute the ordering of simulated regions.
        """
        self.divide_side_n = divide_side_n
        self.overlap = overlap
        self.permute = permute

    def simulate_regions(
        self, full_pixels: Tensor, full_coords: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """See super class for documentation."""
        side_length = full_pixels.shape[0]
        spatial_dim = full_coords.shape[-1]
        rgb_dim = full_pixels.shape[-1]

        side_length_per_region = side_length // self.divide_side_n
        if side_length_per_region * self.divide_side_n != side_length:
            print(
                f"Warning: divide_side_n={self.divide_side_n} does not divide side length {side_length}."
                "Some parts of the image will not belong ot any divided region!"
            )

        # Simulate regions by creating adjacent square patches row-by-row
        pixels_regions = []
        coords_regions = []
        for i in range(self.divide_side_n):
            for j in range(self.divide_side_n):
                pixels_region = full_pixels[
                    i
                    * side_length_per_region : min(
                        (i + 1) * side_length_per_region + self.overlap,
                        side_length,
                    ),
                    j
                    * side_length_per_region : min(
                        (j + 1) * side_length_per_region + self.overlap,
                        side_length,
                    ),
                    :,
                ]
                coords_region = full_coords[
                    i
                    * side_length_per_region : min(
                        (i + 1) * side_length_per_region + self.overlap,
                        side_length,
                    ),
                    j
                    * side_length_per_region : min(
                        (j + 1) * side_length_per_region + self.overlap,
                        side_length,
                    ),
                    :,
                ]
                pixels_regions.append(pixels_region.reshape(-1, rgb_dim))
                coords_regions.append(coords_region.reshape(-1, spatial_dim))

        # Optionally randomly permute the region orderings
        if self.permute:
            permute_regions = torch.randperm(len(pixels_regions))
        else:
            permute_regions = torch.arange(len(pixels_regions))

        pixels_regions_permute = []
        coords_regions_permute = []
        for region in permute_regions:
            pixels_regions_permute.append(pixels_regions[region])
            coords_regions_permute.append(coords_regions[region])

        return pixels_regions_permute, coords_regions_permute


class ImageFitting(Dataset):
    """Dataset that represents a single image for SIREN fitting. One (x, y) data point
    represents one coordinate and its corresponding pixel value in the image.
    """

    def __init__(
        self,
        side_length: int,
        data_split: str,
        id: str,
        region_simulator: RegionSimulator,
    ):
        """Initialize an image fitting dataset.
        Args:
            side_length: Side length of image (i.e. number of pixels on each side).
            data_split: Data split of the BSDS dataset (i.e. 'train' or 'test').
            id: ID of the single image to use.

        """
        super().__init__()
        spatial_dim = 2
        rgb_dim = 3

        # Load full image and its coordinates
        self.full_pixels = get_bsds_tensor_square(side_length, data_split, id)
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Display image
    img = get_bsds_tensor_square(side_length=256, split="test", id="108005")
    print(img.shape)
    fig = plt.figure()
    plt.imshow(img)
    plt.show()
