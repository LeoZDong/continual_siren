from typing import List, Tuple

import torch
from torch import Tensor


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
