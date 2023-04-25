import logging

import torch
from kornia.utils.grid import create_meshgrid3d
from omegaconf import DictConfig

from trainers.hash_freeze_trainer import HashFreezeTrainer
from trainers.simple_trainer import SimpleTrainer
from trainers.simple_trainer_nerf import SimpleTrainerNerf

log = logging.getLogger(__name__)


class NeRFFreezeTrainer(SimpleTrainerNerf, HashFreezeTrainer):
    def __init__(
        self, update_density_bitfield_every_steps: int, warmup_steps: int, **kwargs
    ) -> None:
        HashFreezeTrainer.__init__(self, **kwargs)
        self.update_density_bitfield_every_steps = update_density_bitfield_every_steps
        self.warmup_steps = warmup_steps
        self.num_steps_per_ray = self.network.num_steps_per_ray

        # Re-initialize density bitfield!
        self.network.register_buffer(
            "density_bitfield",
            torch.zeros(
                self.network.bitfield_cascades
                * self.network.density_grid_size**3
                // 8,
                dtype=torch.uint8,
                device=self.network.device,
            ),
        )
        self.network.register_buffer(
            "density_grid",
            torch.zeros(
                self.network.bitfield_cascades,
                self.network.density_grid_size**3,
                device=self.network.device,
            ),
        )

        self.network.register_buffer(
            "density_grid_coords",
            create_meshgrid3d(
                depth=self.network.density_grid_size,
                height=self.network.density_grid_size,
                width=self.network.density_grid_size,
                normalized_coordinates=False,
                dtype=torch.int32,
            )
            .reshape(-1, 3)
            .to(self.network.device),
        )
