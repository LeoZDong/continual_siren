import json
import os
from typing import Dict, List, Tuple

import cv2
import hydra
import numpy as np
import skimage.io as io
import torch
from einops import rearrange
from kornia import create_meshgrid
from torch import Tensor
from torch.utils import data
from torch.utils.data import Dataset
from tqdm import tqdm

from data.region_simulators import RegionSimulator
from utils import create_spheric_poses


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


def get_rays(directions: Tensor, c2w: Tensor):
    """Get ray origin and directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (N, 3) ray directions in camera coordinate
        c2w: (3, 4) or (N, 3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (N, 3), the origin of the rays in world coordinate
        rays_d: (N, 3), the direction of the rays in world coordinate
    """
    if c2w.ndim == 2:
        # Rotate ray directions from camera coordinate to the world coordinate
        rays_d = directions @ c2w[:, :3].T
    else:
        rays_d = rearrange(directions, "n c -> n 1 c") @ rearrange(
            c2w[..., :3], "n a b -> n b a"
        )
        rays_d = rearrange(rays_d, "n 1 c -> n c")
    # Make ray directions unit vectors
    rays_d /= torch.linalg.norm(rays_d, dim=1, keepdim=True)

    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[..., 3].expand_as(rays_d)

    return rays_o, rays_d


class NeRFSyntheticDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str,
        downsample: float,
        region_simulator: RegionSimulator,
    ):
        super().__init__()
        try:
            self.path = os.path.join(hydra.utils.get_original_cwd(), path)
        except:
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

        # Generate a set of poses for video generation
        self.generate_video_poses()
        # Number of frames / poses for generating the test time video
        num_frames_for_video = self.rays_o_video.shape[0]
        self._video_inputs = [
            {"rays_o": self.rays_o_video[i], "rays_d": self.rays_d_video[i]}
            for i in range(num_frames_for_video)
        ]

        # Simulate regions
        self._input_regions, self._output_regions = region_simulator.simulate_regions(
            {"rays_o": self.rays_o, "rays_d": self.rays_d}, self.pixels
        )

        # Set "full data" (combination of rays in all regions)
        self._full_input = {
            "rays_o": self.rays_o.reshape(-1, 3),
            "rays_d": self.rays_d.reshape(-1, 3),
        }
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
            rays_o: (num_frames, h * w, 3) Origins of all rays in all image frames in
                world coordinate.
            rays_d: (num_frames, h * w, 3) Corresponding ray directions.
            pixels: (num_frames, h * w, 3) RGB pixel values of each image frame (flattened).
        """
        self.rays_o = []
        self.rays_d = []
        self.pixels = []

        with open(os.path.join(self.path, f"transforms_{split}.json"), "r") as f:
            frames = json.load(f)["frames"]

        print(f"Loading {len(frames)} {split} images ...")
        heights = []
        for frame in tqdm(frames):
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)[:3, :4]

            # Scale camera-to-world transform matrix
            c2w[:, 1:3] *= -1  # [right up back] to [right down front]

            # NOTE: leaving this here as `ngp_pl` has it. But having scale be 1 makes
            # the cameras all within the [-1, 1] bound (on x and y; within [0, 1] on z).
            # pose_radius_scale = 1.5
            pose_radius_scale = 1
            c2w[:, 3] /= torch.linalg.norm(c2w[:, 3]) / pose_radius_scale

            # Obtain rays in world coordinate
            rays_o, rays_d = get_rays(self.directions, c2w)
            self.rays_o.append(rays_o)
            self.rays_d.append(rays_d)

            # Record corresponding pixel values to these rays
            img_path = os.path.join(self.path, f"{frame['file_path']}.png")
            img = read_image(img_path, (self.img_w, self.img_h))
            self.pixels.append(img)

            # Record average height of this frame
            heights.append(rays_o[:, 2].mean())

        self.rays_o = torch.stack(self.rays_o)  # (num_frames, h * w, 3)
        self.rays_d = torch.stack(self.rays_d)  # (num_frames, h * w, 3)
        self.pixels = torch.stack(self.pixels)  # (num_frames, h * w, 3)
        self.mean_h = sum(heights) / len(heights)

    def generate_video_poses(self):
        c2ws = create_spheric_poses(radius=1.2, mean_h=self.mean_h, n_poses=120)
        c2ws = torch.FloatTensor(c2ws)
        self.rays_o_video = []
        self.rays_d_video = []

        for c2w in c2ws:
            rays_o, rays_d = get_rays(self.directions, c2w)
            self.rays_o_video.append(rays_o)
            self.rays_d_video.append(rays_d)

        self.rays_o_video = torch.stack(self.rays_o_video)
        self.rays_d_video = torch.stack(self.rays_d_video)

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
            {
                "rays_o": self.rays_o[frame_idx, pixel_idx],
                "rays_d": self.rays_d[frame_idx, pixel_idx],
            },
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

    @property
    def video_inputs(self) -> List[Dict[str, Tensor]]:
        return self._video_inputs
