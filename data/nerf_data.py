import json
import os
from typing import Dict, List, Optional, Tuple, Union

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


def read_image(img_path: str, img_wh: Tuple[int, int], blend_a: bool = True) -> Tensor:
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


# @torch.cuda.amp.autocast(dtype=torch.float32)
def get_ray_directions(
    H: Tensor, W: Tensor, K: Tensor, random=False, return_uv=False, flatten=True
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Get ray directions for all pixels in camera coordinate [right down front].
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Args:
        H, W: image height and width
        K: (3, 3) camera intrinsics
        random: whether the ray passes randomly inside the pixel
        return_uv: whether to return uv image coordinates

    Returns: (shape depends on @flatten)
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


# @torch.cuda.amp.autocast(dtype=torch.float32)
def get_rays(directions: Tensor, c2w: Tensor) -> Tensor:
    """Get ray origin and directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Args:
        directions: (N, 3) ray directions in camera coordinate
        c2w: (3, 4) or (N, 3, 4) transformation matrix from camera coordinate to world coordinate

    Returns:
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
        split: Union[str, List],
        img_w: int,
        img_h: int,
        downsample: float,
        use_test_split_for_video: bool,
        region_simulator: RegionSimulator,
        continual: bool,
    ) -> None:
        """Initialize a NeRF synthetic dataset.
        Args:
            continual: If True, __len__ and __getitem__ pick from the current region.
                Else, they pick from the full available dataset.
        """
        super().__init__()
        try:
            self.path = os.path.join(hydra.utils.get_original_cwd(), path)
        except:
            self.path = path
        self.downsample = downsample
        self.region_simulator = region_simulator
        self.continual = continual

        # Fixed values for NeRF Synthetic dataset
        self.img_w = int(img_w * self.downsample)
        self.img_h = int(img_h * self.downsample)

        # Load data and metadata
        # TODO: These do not need to be instance variables
        self.K, self.directions = self.read_intrinsics()

        if isinstance(split, List):
            self.rays_o = []
            self.rays_d = []
            self.pixels = []
            self.poses = []
            for s in split:
                rays_o, rays_d, pixels, poses = self.read_data(s, self.directions)
                self.rays_o.append(rays_o)
                self.rays_d.append(rays_d)
                self.pixels.append(pixels)
                self.poses.append(poses)
            self.rays_o = torch.concat(self.rays_o)
            self.rays_d = torch.concat(self.rays_d)
            self.pixels = torch.concat(self.pixels)
            self.poses = torch.concat(self.poses)
        else:
            self.rays_o, self.rays_d, self.pixels, self.poses = self.read_data(
                split, self.directions
            )

        # Generate a set of poses for video generation
        if use_test_split_for_video:
            rays_o, rays_d, pixels, poses = self.read_data("test", self.directions)
            num_frames_for_video = rays_o.shape[0]
            self._video_inputs = [
                {"rays_o": rays_o[i], "rays_d": rays_d[i]}
                for i in range(num_frames_for_video)
            ]
            self._video_outputs = [pixels[i] for i in range(num_frames_for_video)]
        else:
            rays_o, rays_d = self.generate_video_rays(self.directions, max_height=0.5)
            num_frames_for_video = rays_o.shape[0]
            self._video_inputs = [
                {"rays_o": rays_o[i], "rays_d": rays_d[i]}
                for i in range(num_frames_for_video)
            ]
            self._video_outputs = None

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

    def read_intrinsics(self) -> Tuple[Tensor, Tensor]:
        """Read and store local camera intrinsics as instance variables.

        Returns:
            K: (3, 3, 3) Camera intrinsics matrix.
            directions: (h * w, 3) Directions of rays towards each pixel in camera coordinates.
        """

        # Meta data `camera_angle_x` is the same across splits, so just read from train
        with open(os.path.join(self.path, "transforms_train.json"), "r") as f:
            meta = json.load(f)

        fx = 0.5 * self.img_w / np.tan(0.5 * meta["camera_angle_x"])
        fy = fx

        K = torch.tensor(
            [[fx, 0, self.img_w / 2], [0, fy, self.img_h / 2], [0, 0, 1]],
            dtype=torch.float32,
        )
        directions = get_ray_directions(self.img_h, self.img_w, K)
        return K, directions

    def read_data(
        self, split: str, directions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Read and store image frames and their corresponding poses.
        Args:
            split: Split to use.
            directions: (h * w, 3) Directions of rays towards each pixel in camera coordinates.

        Returns:
            rays_o: (num_frames, h * w, 3) Origins of all rays in all image frames in
                world coordinate.
            rays_d: (num_frames, h * w, 3) Corresponding ray directions.
            pixels: (num_frames, h * w, 3) RGB pixel values of each image frame (flattened).
            poses: (num_frames, 3, 4) Camera pose (camera-to-world transformation matrix)
                of each image frame. This is not used in training, but is handy for
                sanity check where we could make the test video poses the same as
                training poses.
        """
        rays_o = []
        rays_d = []
        pixels = []
        poses = []

        with open(os.path.join(self.path, f"transforms_{split}.json"), "r") as f:
            frames = json.load(f)["frames"]

        print(f"Loading {len(frames)} {split} images ...")
        for frame in tqdm(frames):
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)[:3, :4]

            # Scale camera-to-world transform matrix
            c2w[:, 1:3] *= -1  # [right up back] to [right down front]

            # NOTE: leaving this here as `ngp_pl` has it. But having scale be 1 makes
            # the cameras all within the [-1, 1] bound (on x and y; within [0, 1] on z).
            # If we have scale 1.5, then z is in [0, 1.5].
            # pose_radius_scale = 1.5
            pose_radius_scale = 1
            c2w[:, 3] /= torch.linalg.norm(c2w[:, 3]) / pose_radius_scale
            poses.append(c2w)

            # Obtain rays in world coordinate
            rays_o_frame, rays_d_frame = get_rays(directions, c2w)
            rays_o.append(rays_o_frame)
            rays_d.append(rays_d_frame)

            # Record corresponding pixel values to these rays
            img_path = os.path.join(self.path, f"{frame['file_path']}.png")
            img = read_image(img_path, (self.img_w, self.img_h))
            pixels.append(img)

        rays_o = torch.stack(rays_o)  # (num_frames, h * w, 3)
        rays_d = torch.stack(rays_d)  # (num_frames, h * w, 3)
        pixels = torch.stack(pixels)  # (num_frames, h * w, 3)
        poses = torch.stack(poses)  # (num_frames, 3, 4)

        return rays_o, rays_d, pixels, poses

    def generate_video_rays(
        self, directions: Tensor, max_height: float
    ) -> Tuple[Tensor, Tensor]:
        """Generate novel rays to visualize a test time video. We do not have ground-truth
        pixels corresponding to these rays. Rays have 3 heights evenly spaced in [0, max_height].
        Args:
            directions: (h * w, 3) Directions of rays towards each pixel in camera coordinates.

        Returns:
            rays_o_video: (num_frames, h * w, 3)
            rays_d_video: (num_frames, h * w, 3)
        """
        # Evaluate on 3 different heights
        c2ws = []
        for i in range(3):
            # TODO: Mean height is not a great heuristic to get a good testing height.
            # Figure out a better way?
            # TODO: One way is to also get the mean pitch angle. Then we can use mean
            # height + pitch angle
            pose = torch.FloatTensor(
                create_spheric_poses(
                    radius=1, mean_h=max_height * ((i + 1) / 3), n_poses=10
                )
            )
            pose[:, :, 3] /= torch.linalg.norm(pose[:, :, 3], axis=1, keepdim=True)
            c2ws.append(pose)

        c2ws = torch.concatenate(c2ws)

        rays_o_video = []
        rays_d_video = []

        for c2w in c2ws:
            rays_o, rays_d = get_rays(directions, c2w)
            rays_o_video.append(rays_o)
            rays_d_video.append(rays_d)

        return torch.stack(rays_o_video), torch.stack(rays_d_video)

    def __len__(self) -> int:
        """Dataset size is defined as the total number pixels / rays."""
        if self.continual:
            return self.output.shape[0]
        else:
            return self.full_output.shape[0]

    def __getitem__(self, idx) -> Tuple[Dict[str, Tensor], Tensor]:
        """Given `idx`, we flatten it into `frame_idx` and `pixel_idx` and return
        information about one ray.

        Returns: 2-tuple for input and output
            Input: 'directions': (3,) Direction of ray in camera coordinate.
                   'poses': (3, 4) Pose of camera (camera-to-world transform matrix).
            Output: (3, ) Pixel value of this ray.
        """
        if self.continual:
            return {
                "rays_o": self.input["rays_o"][idx],
                "rays_d": self.input["rays_d"][idx],
            }, self.output[idx]
        else:
            return {
                "rays_o": self.full_input["rays_o"][idx],
                "rays_d": self.full_input["rays_d"][idx],
            }, self.full_output[idx]

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

    @property
    def video_outputs(self) -> Optional[List[Tensor]]:
        """If None, it means that we do not have ground-truth pixels for video rays."""
        return self._video_outputs
