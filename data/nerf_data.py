import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import skimage.io as io
import torch
from kornia import create_meshgrid
from torch import Tensor
from torch.utils import data
from torch.utils.data import Dataset
from tqdm import tqdm

from data.region_simulators import RegionSimulator


def read_image(img_path: str, img_wh: Tuple[int, int], blend_a: bool = True):
    # TODO: Combine this with `image_data.py` function
    img = io.imread(img_path).astype(np.float32) / 255
    if img.shape[2] == 4:  # blend A to RGB
        if blend_a:
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
        else:
            img = img[..., :3] * img[..., -1:]

    img = cv2.resize(img, img_wh)
    img = torch.tensor(img.reshape((-1, 3)), dtype=torch.float32)

    return img


def get_ray_directions(H, W, K, random=False, return_uv=False, flatten=True):
    """Get ray directions for all pixels in camera coordinate [right down front].
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics
        random: whether the ray passes randomly inside the pixel
        return_uv: whether to return uv image coordinates

    Outputs: (shape depends on @flatten)
        directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
        uv: (H, W, 2) or (H*W, 2) image coordinates
    """
    grid = create_meshgrid(H, W, False)[0]  # (H, W, 2)
    u, v = grid.unbind(-1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if random:
        directions = torch.stack(
            [
                (u - cx + torch.rand_like(u)) / fx,
                (v - cy + torch.rand_like(v)) / fy,
                torch.ones_like(u),
            ],
            -1,
        )
    else:  # pass by the center
        directions = torch.stack(
            [(u - cx + 0.5) / fx, (v - cy + 0.5) / fy, torch.ones_like(u)], -1
        )
    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1, 2)

    if return_uv:
        return directions, grid
    return directions


class NeRFSyntheticDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str,
        downsample: float,
        region_simulator: RegionSimulator,
    ):
        super().__init__()
        self.path = path
        self.split = split
        self.downsample = downsample
        self.region_simulator = region_simulator

        # Fixed values for NeRF Synthetic dataset
        self.img_w = int(800 * self.downsample)
        self.img_h = int(800 * self.downsample)

        # Load data and metadata
        self.read_intrinsics()
        self.read_data(split)

        # Simulate regions
        self._input_regions, self._output_regions = region_simulator.simulate_regions(
            {"directions": self.directions, "poses": self.poses}, self.pixels
        )

        # Set "full data" (combination of rays in all regions)
        num_frames = self.pixels.shape[0]
        num_rays_per_frame = self.pixels.shape[1]
        full_directions = self.directions.unsqueeze(0).expand(
            num_frames, num_rays_per_frame, 3
        )
        full_directions = full_directions.reshape(-1, 3)
        full_poses = self.poses.unsqueeze(1).expand(
            num_frames, num_rays_per_frame, 3, 4
        )
        full_poses = full_poses.reshape(-1, 3, 4)
        self._full_input = {"directions": full_directions, "poses": full_poses}
        self._full_output = self.pixels.reshape(-1, 3)

        # Initialize current region to be the first region
        self.set_cur_region(0)

    def set_cur_region(self, region: int) -> None:
        self._cur_region = region
        self._input = self._input_regions[region]
        self._output = self._output_regions[region]

    def read_intrinsics(self):
        """Read and store local camera intrinsics as instance variables.
        Instance variables:
            K: (3, 3, 3) Camera intrinsics matrix.
            directions: (h * w, 3) Directions of rays towards each pixel in camera coordinates.
            img_wh: 2-tuple of image height and width.
        """
        # Meta data `camera_angle_x` is the same across splits, so just read from train
        with open(os.path.join(self.path, "transforms_train.json"), "r") as f:
            meta = json.load(f)

        fx = 0.5 * self.img_w / np.tan(0.5 * meta["camera_angle_x"])
        fy = fx

        self.K = torch.tensor(
            [[fx, 0, self.img_w / 2], [0, fy, self.img_h / 2], [0, 0, 1]],
            dtype=torch.float32,
        )
        self.directions = get_ray_directions(self.img_h, self.img_w, self.K)

    def read_data(self, split):
        """Read and store image frames and their corresponding poses.
        Instance variables:
            pixels: (num_frames, h * w, 3) RGB pixel values of each image frame (flattened).
            poses: (num_frames, 3, 4) Pose matrix (camera-to-world transformation matrix)
                for each image frame.
        """
        self.pixels = []
        self.poses = []

        with open(os.path.join(self.path, f"transforms_{split}.json"), "r") as f:
            frames = json.load(f)["frames"]

        print(f"Loading {len(frames)} {split} images ...")
        for frame in tqdm(frames):
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)[:3, :4]

            # Scale camera-to-world transform matrix
            c2w[:, 1:3] *= -1  # [right up back] to [right down front]
            pose_radius_scale = 1.5
            c2w[:, 3] /= torch.linalg.norm(c2w[:, 3]) / pose_radius_scale

            self.poses.append(c2w)

            img_path = os.path.join(self.path, f"{frame['file_path']}.png")
            img = read_image(img_path, (self.img_w, self.img_h))
            self.pixels.append(img)

        self.pixels = torch.stack(self.pixels)  # (num_frames, h * w, 3)
        self.poses = torch.stack(self.poses)  # (num_frames, 3, 4)

    def __len__(self):
        """Dataset size is defined as the total number pixels / rays."""
        return self.pixels.shape[0] * self.pixels.shape[1]

    def __getitem__(self, idx):
        """Given `idx`, we flatten it into `frame_idx` and `pixel_idx` and return
        information about one ray.

        Returns: 2-tuple for input and output
            Input: 'directions': (3,) Direction of ray in camera coordinate.
                   'poses': (3, 4) Pose of camera (camera-to-world transform matrix).
            Output: (3, ) Pixel value of this ray.
        """
        frame_idx = idx // self.pixels.shape[1]
        pixel_idx = idx % self.pixels.shape[1]

        return (
            {"directions": self.directions[pixel_idx], "poses": self.poses[frame_idx]},
            self.pixels[frame_idx, pixel_idx],
        )

    @property
    def cur_region(self) -> int:
        return self._cur_region

    @property
    def num_regions(self) -> int:
        return len(self._input_regions)

    @property
    def input(self) -> Tensor:
        return self._input

    @property
    def output(self) -> Tensor:
        return self._output

    @property
    def full_input(self) -> Dict[str, Tensor]:
        return self._full_input

    @property
    def full_output(self) -> Tensor:
        return self._full_output

    @property
    def input_regions(self) -> List:
        return self._input_regions

    @property
    def output_regions(self) -> List:
        return self._output_regions


if __name__ == "__main__":
    dataset = NeRFSyntheticDataset("datasets/nerf_synthetic/lego", "train", 1)
    print(dataset[0])
    loader = data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
    )
    batch = next(iter(loader))
