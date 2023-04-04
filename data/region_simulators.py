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
                Each region has shape (num_points, spatial_dim).
            pixels_regions: List of pixels corresponding to each coordinate region.
                Each region has shape (num_points, value_dim).
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
    """Random region simulator that produces rectangular patches of size uniformly
    distributed in [`region_size_min`, `region_size_max`] (as int, since "size" is the
    number of pixels on one side of the patch). A region contains ground-truth
    coordinates. We ensure that all regions together cover the entire image.
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
        """Repeatedly call `_simulate_regions` to simulate new regions until the entire
        image is covered.

        See `RegularRegionSimulator` for more documentation.
        """
        max_attempts = 10
        for _ in range(max_attempts):
            coords_regions, pixels_regions, all_covered = self._simulate_regions(
                full_coords, full_pixels
            )
            if all_covered:
                return coords_regions, pixels_regions

        raise RuntimeError(
            f"Region simulator cannot cover the full image using {self.num_regions} regions after {max_attempts} trials! Consider increasing `num_regions` or the region size!"
        )

    def _simulate_regions(
        self, full_coords: Tensor, full_pixels: Tensor
    ) -> Tuple[List[Tensor], List[Tensor], bool]:
        """Simulate `self.num_regions` regions and also keep track of whether the list
        of simulated regions cover the entire image.
        """
        # Keep track of coverage (re-initialize when we do a new round of simulation)
        coverage = torch.zeros(
            (full_coords.shape[0], full_coords.shape[1]), dtype=torch.bool
        )

        coords_regions = []
        pixels_regions = []
        spatial_dim = full_coords.shape[2]
        value_dim = full_pixels.shape[2]

        for _ in range(self.num_regions):
            # Pick the size of the region
            region_w = np.random.randint(
                low=self.region_size_min, high=self.region_size_max + 1
            )
            region_h = np.random.randint(
                low=self.region_size_min, high=self.region_size_max + 1
            )

            # Randomly sample an **uncovered** point as region center if not yet all covered
            # NOTE: This is not iid sampling but makes it easier to cover the whole grid
            if not torch.all(coverage):
                not_covered = torch.nonzero(~coverage)
                center = not_covered[np.random.randint(low=0, high=len(not_covered))]
            else:
                center = np.random.randint(
                    low=0, high=full_coords.shape[0], size=(spatial_dim,)
                )

            # TODO: We may get ill-formed regions with length 0 with low probability
            # While this is unlikely to happen, maybe add a guard in the future?
            low_w = max(center[0] - region_w // 2, 0)
            high_w = min(low_w + region_w, full_coords.shape[0])
            low_h = max(center[1] - region_h // 2, 0)
            high_h = min(low_h + region_h, full_coords.shape[1])

            coords = full_coords[low_w:high_w, low_h:high_h, :].reshape(-1, spatial_dim)
            pixels = full_pixels[low_w:high_w, low_h:high_h, :].reshape(-1, value_dim)
            coverage[low_w:high_w, low_h:high_h] = True

            coords_regions.append(coords)
            pixels_regions.append(pixels)

        all_covered = torch.all(coverage)
        return coords_regions, pixels_regions, all_covered


class RegularNeRFRegionSimulator(RegionSimulator):
    def __init__(
        self,
        divide_side_n: int,
        aabb_min: float,
        aabb_max: float,
    ):
        self.divide_side_n = divide_side_n
        self.aabb_min = aabb_min
        self.aabb_max = aabb_max

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

        # Gather frames for each region
        num_frames = full_output.shape[0]
        side_len = self.aabb_max - self.aabb_min
        for i in range(self.divide_side_n):
            for j in range(self.divide_side_n):
                x_bound = [
                    self.aabb_min + i * side_len / self.divide_side_n,
                    self.aabb_min + (i + 1) * side_len / self.divide_side_n,
                ]
                y_bound = [
                    self.aabb_min + j * side_len / self.divide_side_n,
                    self.aabb_min + (j + 1) * side_len / self.divide_side_n,
                ]

                input = {"rays_o": [], "rays_d": []}
                output = []
                # This is not very efficient, but okay for a small number of frames
                for frame_idx in range(num_frames):
                    # Use the first ray's xy as the frame's xy
                    x, y = full_input["rays_o"][frame_idx][0][:2].numpy()
                    if (
                        x > x_bound[0]
                        and x < x_bound[1]
                        and y > y_bound[0]
                        and y < y_bound[1]
                    ):
                        input["rays_o"].append(full_input["rays_o"][frame_idx])
                        input["rays_d"].append(full_input["rays_d"][frame_idx])
                        output.append(full_output[frame_idx])
                input["rays_o"] = torch.concat(input["rays_o"])
                input["rays_d"] = torch.concat(input["rays_d"])
                output = torch.concat(output)
                input_regions.append(input)
                output_regions.append(output)

        return input_regions, output_regions
