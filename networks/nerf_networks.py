import time
from typing import Tuple

import numpy as np
import torch
import vren  # Compiled CUDA backend for volumetric rendering
from einops import rearrange
from kornia.utils.grid import create_meshgrid3d
from torch import Tensor, nn

from networks.hash_networks import HashNet
from networks.misc_networks import SHEncoder, TruncExp
from networks.nerf_cuda_rendering import render_rays_test, render_rays_train
from networks.nerf_custom_functions import RayAABBIntersector
from utils import near_far_from_aabb


class NeRFNetwork(nn.Module):
    """NeRF network that, given a batch of coordinates and viewing directions, return
    the densities and colors at these locations.
    """

    def __init__(
        self,
        density_net: HashNet,  # Recursively instantiated
        direction_encoder: SHEncoder,  # Recursively instantiated
        color_hidden_features: int,
        color_hidden_layers: int,
        color_out_features: int,
        color_out_activation: nn.Module,  # Recursively instantiated
        **kwargs,
    ) -> None:
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
        # TODO: Does this work? Is this necessary?
        self.nets = nn.ModuleList(
            [self.density_net, self.direction_encoder, self.color_net]
        )

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
        density_out = self.density_net(coords)
        density = TruncExp.apply(density_out[:, 0])
        # TODO: Does the first dim of `density_out` also gets TruncExp applied?
        return density, density_out

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
        directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True)
        directions = (directions + 1) / 2
        direction_features = self.direction_encoder(directions)
        input = torch.cat([direction_features, density_out], dim=1)
        return self.color_net(input)

    def forward(
        self, coords: Tensor, directions: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        density, density_out = self.density(coords)
        color = self.color(directions, density_out)
        return density, color

    def to(self, device: torch.device, **kwargs):
        """Convert the network to target `device`."""
        for net in self.nets:
            net.to(device)
        super().to(device, **kwargs)


class NeRFRenderer(nn.Module):
    """NeRF renderer that performs volumetric rendering by querying a NeRF network.
    Given a batch of rays, return the rendered color for each ray.
    """

    def __init__(
        self,
        grid_min: float,
        grid_max: float,
        density_scale: float,
        min_near: float,
        num_steps_per_ray: int,
        optimized_march_cuda: bool,
        nerf_network: NeRFNetwork,  # Recursively instantiated
        **kwargs,
    ) -> None:
        """Initialize a NeRF renderer by recording info about the scene setup.
        Args:
            grid_min: Minimum bound of the 3D grid (axis-aligned bounding box).
            grid_max: Maximum bound of the 3D grid (axis-aligned bounding box).
            density_scale: >1 number to scale output density and delta. Makes the
                rendering sharper.
            min_near: Minimum near intersection time of a ray with aabb.
            num_steps_per_ray: Number of steps to sample along the ray to render.
            optimized_march_cuda: Whether to use optimized march (CUDA only).
        """
        super().__init__()

        self.grid_min = grid_min
        self.grid_max = grid_max
        self.density_scale = density_scale
        self.min_near = min_near
        self.num_steps_per_ray = num_steps_per_ray
        self.optimized_march_cuda = optimized_march_cuda

        # TODO: Register parameters from the NeRF network?
        super().add_module("nerf_network", nerf_network)

        # Prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        aabb_train = torch.tensor(
            [grid_min, grid_min, grid_min, grid_max, grid_max, grid_max],
            dtype=torch.float32,
        )
        aabb_infer = aabb_train.clone()
        self.register_buffer("aabb_train", aabb_train)
        self.register_buffer("aabb_infer", aabb_infer)

        #### Setup for optimized CUDA ray march ####
        self.bitfield_scale = 1
        self.bitfield_cascades = max(
            1 + int(np.ceil(np.log2(2 * self.bitfield_scale))), 1
        )
        self.density_grid_size = 128

        self.register_buffer(
            "density_bitfield",
            torch.zeros(
                self.bitfield_cascades * self.density_grid_size**3 // 8,
                dtype=torch.uint8,
            ),
        )
        self.register_buffer(
            "density_grid",
            torch.ones(self.bitfield_cascades, self.density_grid_size**3),
        )

        self.register_buffer(
            "density_grid_coords",
            create_meshgrid3d(
                depth=self.density_grid_size,
                height=self.density_grid_size,
                width=self.density_grid_size,
                normalized_coordinates=False,
                dtype=torch.int32,
            ).reshape(-1, 3),
        )

    def forward(self, rays_o: Tensor, rays_d: Tensor, **kwargs) -> Tensor:
        """Forward pass to render the pixel values from a batch of rays."""
        pixels, depths = self.render(rays_o, rays_d, **kwargs)
        return pixels

    def _render_simple_march(
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
        density, density_out = self.nerf_network.density(xyzs.reshape(-1, 3))
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

        color[mask] = self.nerf_network.color(
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

    @torch.cuda.amp.autocast()
    def _render_optimized_march_cuda(
        self,
        rays_o: Tensor,
        rays_d: Tensor,
        **kwargs,
    ):
        """Render a batch of rays using accelerated and optimized ray marching. Entirely
        implemented in CUDA and can only be called when device is cuda.
        """
        # FIXME: For some reason, using optimized algorithm leads to some memory leakage (?)
        # where I get oom at around step 350 (also does not happen when I don't update bitfield).
        torch.cuda.empty_cache()

        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        center = (
            (self.grid_min + self.grid_max) / 2 * torch.ones((1, 3), device=self.device)
        )
        half_size = (
            (self.grid_max - self.grid_min) / 2 * torch.ones((1, 3), device=self.device)
        )

        _, hits_t, _ = RayAABBIntersector.apply(rays_o, rays_d, center, half_size, 1)
        hits_t[
            (hits_t[:, 0, 0] >= 0) & (hits_t[:, 0, 0] < self.min_near), 0, 0
        ] = self.min_near

        if self.training:
            render_func = render_rays_train
        else:
            render_func = render_rays_test
            # NOTE: test rendering is significantly slower for some reason? Use train rendering for now???
            # render_func = render_rays_train

        results = render_func(
            self.nerf_network,
            rays_o,
            rays_d,
            hits_t,
            self.density_bitfield,
            self.bitfield_cascades,
            self.bitfield_scale,
            self.density_grid_size,
            max_samples=self.num_steps_per_ray,
            **kwargs,
        )
        for k, v in results.items():
            if kwargs.get("to_cpu", False):
                v = v.cpu()
                if kwargs.get("to_numpy", False):
                    v = v.numpy()
            results[k] = v

        # We only return the RGB and depths!
        return results["rgb"], results["depth"]

    def render(
        self, rays_o, rays_d, max_ray_batch: int = 1024, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Render rays. Stage the rendering into batches if too many rays.
        Args:
            rays_o: (num_rays, 3) Origins of the rays.
            rays_d: (num_rays, 3) Directions of the rays.
            max_ray_batch: If bsz > max_ray_batch, stage the rendering into batches.

        Returns:
            rgb: (num_rays, 3) Rendered RGB value for each ray.
            depth: (num_rays, ) Rendered depth for each ray.
        """
        num_rays = rays_o.shape[0]
        device = rays_o.device

        _render = (
            self._render_optimized_march_cuda
            if self.optimized_march_cuda
            else self._render_simple_march
        )

        if num_rays > max_ray_batch:
            image = torch.empty((num_rays, 3), dtype=torch.float32, device=device)
            depth = torch.empty((num_rays,), dtype=torch.float32, device=device)

            i = 0
            while i < num_rays:
                image_batch, depth_batch = _render(
                    rays_o[i : min(i + max_ray_batch, num_rays)],
                    rays_d[i : min(i + max_ray_batch, num_rays)],
                    num_steps=self.num_steps_per_ray,
                    perturb=True,
                    **kwargs,
                )
                image[i : min(i + max_ray_batch, num_rays)] = image_batch
                depth[i : min(i + max_ray_batch, num_rays)] = depth_batch
                i += max_ray_batch

            return image, depth

        else:
            return _render(rays_o, rays_d, num_steps=self.num_steps_per_ray, **kwargs)

    # ======== Methods used by optimized ray marching ========
    @torch.no_grad()
    def get_all_cells(self):
        """Get all cells from the density grid.

        Returns:
            cells: list (of length self.bitfield_cascades) of indices and coords
                selected at each cascade
        """
        indices = vren.morton3D(self.density_grid_coords).long()
        cells = [(indices, self.density_grid_coords)] * self.bitfield_cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """Sample both M uniform and occupied cells (per cascade) occupied cells are
        sample from cells with density > @density_threshold.

        Returns:
            cells: list (of length self.bitfield_cascades) of indices and coords
                selected at each cascade
        """
        cells = []
        for c in range(self.bitfield_cascades):
            # uniform cells
            coords1 = torch.randint(
                self.density_grid_size,
                (M, 3),
                dtype=torch.int32,
                device=self.density_grid.device,
            )
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c] > density_threshold)[:, 0]
            if len(indices2) > 0:
                rand_idx = torch.randint(
                    len(indices2), (M,), device=self.density_grid.device
                )
                indices2 = indices2[rand_idx]
            coords2 = vren.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=64**3):
        """Mark the cells that aren't covered by the cameras with density -1. This is
        only executed once before training starts.
        Args:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        N_cams = poses.shape[0]
        self.count_grid = torch.zeros_like(self.density_grid)
        w2c_R = rearrange(poses[:, :3, :3], "n a b -> n b a")  # (N_cams, 3, 3)
        w2c_T = -w2c_R @ poses[:, :3, 3:]  # (N_cams, 3, 1)
        cells = self.get_all_cells()
        for c in range(self.bitfield_cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i : i + chunk] / (self.density_grid_size - 1) * 2 - 1
                s = min(2 ** (c - 1), self.bitfield_scale)
                half_grid_size = s / self.density_grid_size
                xyzs_w = (xyzs * (s - half_grid_size)).T  # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T  # (N_cams, 3, chunk)
                uvd = K @ xyzs_c  # (N_cams, 3, chunk)
                uv = uvd[:, :2] / uvd[:, 2:]  # (N_cams, 2, chunk)
                in_image = (
                    (uvd[:, 2] >= 0)
                    & (uv[:, 0] >= 0)
                    & (uv[:, 0] < img_wh[0])
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < img_wh[1])
                )
                covered_by_cam = (
                    uvd[:, 2] >= self.min_near
                ) & in_image  # (N_cams, chunk)
                # if the cell is visible by at least one camera
                self.count_grid[c, indices[i : i + chunk]] = count = (
                    covered_by_cam.sum(0) / N_cams
                )

                too_near_to_cam = (uvd[:, 2] < self.min_near) & in_image  # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count > 0) & (~too_near_to_any_cam)
                self.density_grid[c, indices[i : i + chunk]] = torch.where(
                    valid_mask, 0.0, -1.0
                )

    @torch.no_grad()
    def update_density_grid(
        self, density_threshold, warmup=False, decay=0.95, erode=False
    ):
        density_grid_tmp = torch.zeros_like(self.density_grid)
        if warmup:  # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(
                self.density_grid_size**3 // 4, density_threshold
            )
        # infer sigmas
        for c in range(self.bitfield_cascades):
            indices, coords = cells[c]
            s = min(2 ** (c - 1), self.bitfield_scale)
            half_grid_size = s / self.density_grid_size
            xyzs_w = (coords / (self.density_grid_size - 1) * 2 - 1) * (
                s - half_grid_size
            )
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w) * 2 - 1) * half_grid_size
            density_grid_tmp[c, indices] = self.nerf_network.density(xyzs_w)[0]

        self.density_grid = torch.where(
            self.density_grid < 0,
            self.density_grid,
            torch.maximum(self.density_grid * decay, density_grid_tmp),
        )
        mean_density = self.density_grid[self.density_grid > 0].mean().item()

        vren.packbits(
            self.density_grid,
            min(mean_density, density_threshold),
            self.density_bitfield,
        )

    def to(self, device: torch.device, **kwargs):
        self.device = device
        self.nerf_network.to(device, **kwargs)
        super().to(device, **kwargs)
