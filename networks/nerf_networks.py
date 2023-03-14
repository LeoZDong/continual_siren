import time
from typing import Tuple

import numpy as np
import torch
from torch import Tensor, nn

from networks.hash_networks import HashNet
from networks.misc_networks import SHEncoder, TruncExp

from utils import near_far_from_aabb


class NeRFRenderer(nn.Module):
    """NeRF renderer base that performs volumetric rendering by querying a density
    function and a color function.
    """

    def __init__(
        self,
        grid_min: float,
        grid_max: float,
        density_scale: float,
        min_near: float,
        num_steps_per_ray: int,
        **kwargs,
    ):
        """Initialize a base NeRF renderer by recording info about the scene setup.
        Args:
            grid_min: Minimum bound of the 3D grid (axis-aligned bounding box).
            grid_max: Maximum bound of the 3D grid (axis-aligned bounding box).
            density_scale: >1 number to scale output density and delta. Makes the
                rendering sharper.
            min_near: Minimum near intersection time of a ray with aabb.
            num_steps_per_ray: Number of steps to sample along the ray to render.
        """
        super().__init__()

        self.grid_min = grid_min
        self.grid_max = grid_max
        self.density_scale = density_scale
        self.min_near = min_near
        self.num_steps_per_ray = num_steps_per_ray

        # Prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        aabb_train = torch.tensor(
            [grid_min, grid_min, grid_min, grid_max, grid_max, grid_max],
            dtype=torch.float32,
        )
        aabb_infer = aabb_train.clone()
        self.register_buffer("aabb_train", aabb_train)
        self.register_buffer("aabb_infer", aabb_infer)

    def forward(self, rays_o: Tensor, rays_d: Tensor) -> Tensor:
        """Forward pass to render the pixel values from a batch of rays."""
        raise NotImplementedError()

    def density(self, coords: Tensor) -> Tuple[Tensor, Tensor]:
        """Query the density network. Given a batch of (world) coordinates, return their
        density values.
        Args:
            coords: (bsz, 3) Batch of real world coordinates.

        Returns:
            density: (bsz, ) Density of the input coordinates.
            density_out: (bsz, dim) Raw output of the density network that contains
                additional features to be used as input to the color network.
        """
        raise NotImplementedError()

    def color(self, directions: Tensor, density_out: Tensor) -> Tensor:
        """Query the color network. Given a batch of viewing directions and spatial
        features (output of density network from input coordinates), return the RGB values.
        Args:
            directions: (bsz, 3) Batch of viewing directions.
            density_out: (bsz, dim) Raw output of the density network that encodes a
                batch of input coordinates.

        Returns:
            color: (bsz, 3) RGB colors of the input coordinates.
        """
        raise NotImplementedError()

    def _render(
        self,
        rays_o: Tensor,
        rays_d: Tensor,
        num_steps: Tensor,
        perturb: bool,
        **kwargs,
    ):
        """Render a batch of rays using simple ray marching by uniform sampling.
        Entirely implemented in slow but flexible Pytorch operations.
        Args:
            rays_o: (bsz, 3) Origins of the batch of rays.
            rays_d: (bsz, 3) Directions of the batch of rays (unit vectors).
            num_steps: Number of steps to take along each ray from near intersection
                point to far intersection point to calculate the final RGB value.
            perturb: Whether to perturb the samples along each ray.

        Returns:
            depth: (bsz, ) Rendered depth for each ray.
            rgb: (bsz, 3) Rendered RGB value for each ray.
        """
        bsz = rays_o.shape[0]
        device = rays_o.device

        # Choose bounding box
        aabb = self.aabb_train if self.training else self.aabb_infer

        #### Sample points along each ray ####
        t = time.time()
        nears, fars = near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        # print(f"aabb time: {time.time() - t}")

        # Depth values in camera coordinate ("march step size" from ray origin in ray direction)
        # Shape: (bsz, num_steps)
        t = time.time()
        z_vals = (
            torch.linspace(0, 1, num_steps, device=device)
            .unsqueeze(0)
            .expand((bsz, num_steps))
        )
        # Scale `z_vals` (depth) based on near and far intersection times. If the closer
        # the near and far intersections are, the denser the sampling points.
        z_vals = nears + (fars - nears) * z_vals

        # Expected distance between two sample points
        sample_distance = (fars - nears) / num_steps

        # Perturb depth values
        if perturb:
            noise = (torch.rand(z_vals.shape, device=device) - 0.5) * sample_distance
            z_vals += noise
            # z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # Generate sample points (shape: (bsz, num_steps, 3))
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # Clamp to aabb bound
        # print(f"Ray generation time: {time.time() - t}")

        #### Render each ray by querying sample points along each ray ####
        ## Compute density for each point ##
        t = time.time()
        density, density_out = self.density(xyzs.reshape(-1, 3))
        density = density.reshape(bsz, num_steps)
        # print(f"Density time: {time.time() - t}")

        ## Compute alpha channel and weights ##
        t = time.time()
        # Actual distance between each pair of sample points
        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T-1]
        deltas = torch.cat(
            [deltas, sample_distance * torch.ones_like(deltas[..., :1])], dim=-1
        )
        alphas = 1 - torch.exp(
            -deltas * self.density_scale * density.squeeze(-1)
        )  # [N, T+t]
        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1
        )  # [N, T+t+1]

        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T+t]
        # print(f"Alpha time: {time.time() - t}")

        ## Compute color for each point ##
        t = time.time()
        directions = rays_d.reshape(-1, 1, 3).expand_as(xyzs).reshape(-1, 3)
        mask = weights > 1e-4  # hard coded
        mask = mask.flatten()
        color = torch.zeros_like(directions)

        color[mask] = self.color(
            directions=directions[mask], density_out=density_out[mask]
        )
        color = color.reshape(bsz, num_steps, 3)
        # print(f"Color time: {time.time() - t}")

        ## Compute color for the entire ray ##
        t = time.time()
        pixels = torch.sum(weights.unsqueeze(-1) * color, dim=-2)  # (bsz, 3)
        # Fill in the remaining color of a ray as white
        pixels += 1 - weights.sum(1, keepdim=True)

        ## Compute depth for the entire ray ##
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)
        # print(f"Compose time: {time.time() - t}")

        return pixels, depth

class NeRFNetwork(NeRFRenderer):
    def __init__(
        self,
        density_net: HashNet,  # Recursively instantiated
        direction_encoder: SHEncoder,  # Recursively instantiated
        color_hidden_features: int,
        color_hidden_layers: int,
        color_out_features: int,
        color_out_activation: nn.Module,  # Recursively instantiated
        **kwargs,
    ):
        super().__init__(**kwargs)

        #### Density network is a HashNet (hash encoding + MLP) ####
        # Points (bsz, 3) -> density + features (bsz, 1 + 15)
        self.density_net = density_net

        #### View direction encoder ####
        self.direction_encoder = direction_encoder

        #### Color network is a simple MLP ####
        color_in_features = direction_encoder.out_dim + density_net.out_dim
        self.color_net = []
        self.color_net.append(nn.Linear(color_in_features, color_hidden_features))
        self.color_net.append(nn.ReLU())
        for i in range(color_hidden_layers):
            self.color_net.append(
                nn.Linear(color_hidden_features, color_hidden_features)
            )
            self.color_net.append(nn.ReLU())
        self.color_net.append(nn.Linear(color_hidden_features, color_out_features))
        self.color_net.append(color_out_activation)
        self.color_net = nn.Sequential(*self.color_net)

        # Register as module list
        self.nets = nn.ModuleList(
            [self.density_net, self.direction_encoder, self.color_net]
        )

    def density(self, coords: Tensor) -> Tuple[Tensor, Tensor]:
        """Query the density network. See super class documentation."""
        density_out = self.density_net(coords)
        density = TruncExp.apply(density_out[:, 0])
        # TODO: Does the first dim of `density_out` also gets TruncExp applied?
        return density, density_out

    def color(self, directions: Tensor, density_out: Tensor) -> Tensor:
        """Query the color network. See super class documentation."""
        directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True)
        directions = (directions + 1) / 2
        direction_features = self.direction_encoder(directions)
        input = torch.cat([direction_features, density_out], dim=1)
        return self.color_net(input)

    def forward(self, rays_o: Tensor, rays_d: Tensor) -> Tensor:
        """Forward pass just renders the batch of input rays."""
        pixels, depth = self.render(
            rays_o,
            rays_d,
            max_ray_batch=1024,
            num_steps=self.num_steps_per_ray,
            perturb=True,
        )

        return pixels

    def to(self, device: torch.device, **kwargs):
        """Convert the network to target `device`."""
        for net in self.nets:
            net.to(device)
        super().to(device, **kwargs)
