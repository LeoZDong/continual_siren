import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from torch import Tensor


class RegionSimulator:
    def simulate_regions(self, full_input: Any, full_output: Any) -> Tuple[List, List]:
        """Simulate regions.
        Args:
            full_input: Full set of input data.
            full_output: Full set of output data.

        Returns:
            input_regions: List of input regions.
            output_regions: Corresponding output regions.
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
        self, full_coords: Tensor, full_pixels: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Simulate regions for image fitting.
        Args:
            full_coords: (side_length, side_length, spatial_dim) Full set of ground
                truth coordinates.
            full_pixels: (side_length, side_length, value_dim) Full set of ground truth
                pixels corresponding to `full_coords`

        Returns:
            coords_regions: List of coordinate regions.
            pixels_regions: List of pixels corresponding to each coordinate region.
        """
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
        coords_regions = []
        pixels_regions = []
        for i in range(self.divide_side_n):
            for j in range(self.divide_side_n):
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
                coords_regions.append(coords_region.reshape(-1, spatial_dim))
                pixels_regions.append(pixels_region.reshape(-1, rgb_dim))

        # Optionally randomly permute the region orderings
        permute_regions = list(range(len(coords_regions)))
        if self.permute:
            random.shuffle(permute_regions)

        coords_regions_permute = []
        pixels_regions_permute = []
        for region in permute_regions:
            coords_regions_permute.append(coords_regions[region])
            pixels_regions_permute.append(pixels_regions[region])

        return coords_regions_permute, pixels_regions_permute


class RandomRegionSimulator(RegionSimulator):
    """Random region simulator that produces square patches of size uniformly distributed
    in [`region_size_min`, `region_size_max`]. Coordinates inside regions are continous
    coordinates that might not match ground-truth in `full_coords`, in which case the
    corresponding pixel values are interpolated.
    """

    def __init__(
        self,
        num_regions: int,
        region_size_min: float,
        region_size_max: float,
    ):
        self.num_regions = num_regions
        self.region_size_min = region_size_min
        self.region_size_max = region_size_max

    def simulate_regions(
        self, full_coords: Tensor, full_pixels: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """See `RegularRegionSimulator` for documentation."""
        coords_regions = []
        pixels_regions = []

        axes = (full_coords[:, 0, 0].numpy(), full_coords[0, :, 1].numpy())
        interpolate = RegularGridInterpolator(
            axes, full_pixels.numpy(), method="linear"
        )

        coords_max = full_coords.max().item()
        coords_min = full_coords.min().item()
        spatial_dim = full_coords.shape[-1]
        density = full_coords.shape[0] / (coords_max - coords_min)

        for _ in range(self.num_regions):
            # Pick the size (side length) of the region
            region_size = np.random.uniform(
                low=self.region_size_min, high=self.region_size_max
            )
            # Pick the bottom left corner of the region
            bottom_left = (
                np.random.rand(spatial_dim) * (coords_max - coords_min - region_size)
                + coords_min
            )

            # Randomly select coords within region and interpolate
            num_coords = int((region_size * density) ** spatial_dim)
            coords = np.random.uniform(
                low=bottom_left,
                high=bottom_left + region_size,
                size=(num_coords, spatial_dim),
            )
            pixels = interpolate(coords)
            coords_regions.append(torch.tensor(coords, dtype=full_coords.dtype))
            pixels_regions.append(torch.tensor(pixels, dtype=full_pixels.dtype))

        return coords_regions, pixels_regions


class NeRFRegionSimulator(RegionSimulator):
    def __init__(
        self,
        num_images_per_region: int,
        img_h: int,
        img_w: int,
    ):
        self.num_images_per_region = num_images_per_region
        # Not used for not; needed later when one image is further divided into regions
        self.img_h = img_h
        self.img_w = img_w

    def simulate_regions(
        self, full_input: Dict[str, Tensor], full_output: Tensor
    ) -> Tuple[List[Dict[str, Tensor]], List[Tensor]]:
        """Simulate regions for NeRF training. One region contains all rays in one image.
        Args:
            full_input: Dictionary containing:
                'rays_o': (num_frames, h * w, 3) Origins of all rays in all image frames.
                'rays_d': (num_frames, h * w, 3) Corresponding ray directions.
            full_output: (num_frames, h * w, 3) Pixel values of all rays in all image frames.

        Returns:
            input_regions: List where each item is a dictionary containing:
                'rays_o': (h * w, 3) Origins of rays in this region.
                'rays_d': (h * w, 3) Directions of rays in this region.
            output_regions: List where each item is:
                (h * w, 3) Pixel values of each ray in this region.
        """
        input_regions = []
        output_regions = []

        num_frames = full_input["rays_o"].shape[0]
        for frame_idx in range(num_frames):
            # Pose is expanded to one pose per ray, even though poses for all rays are
            # the same. Using `expand` does not allocate more memory.
            input = {
                "rays_o": full_input["rays_o"][frame_idx],
                "rays_d": full_input["rays_d"][frame_idx],
            }
            output = full_output[frame_idx]
            input_regions.append(input)
            output_regions.append(output)

        return input_regions, output_regions
