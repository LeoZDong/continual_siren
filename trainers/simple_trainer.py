import math
import os
from typing import Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter


class SimpleTrainer:
    def __init__(self, cfg: DictConfig, **kwargs):
        self.cfg = cfg
        self.siren = instantiate(cfg.network)
        self.dataset = instantiate(cfg.data)
        self.device = (
            torch.device("cpu")
            if not torch.cuda.is_available() or not cfg.trainer.cuda_if_available
            else torch.device("cuda")
        )
        self.optimizer = instantiate(
            self.cfg.trainer.optimizer, params=self.siren.parameters()
        )
        self._tb_writer = SummaryWriter(os.getcwd())

    def train(self) -> None:
        total_steps = self.cfg.trainer.total_steps

        # Prepare data
        dataloader = data.DataLoader(
            self.dataset, batch_size=len(self.dataset), shuffle=False
        )
        model_input, ground_truth = next(iter(self.dataset))
        model_input, ground_truth = model_input.to(self.device), ground_truth.to(
            self.device
        )

        step = 0
        while step <= total_steps:
            model_output, coords = self.siren(model_input)
            loss = ((model_output - ground_truth) ** 2).mean()

            # Logging training loss
            if step % self.cfg.trainer.summary_every_steps == 0:
                self._tb_writer.add_scalar(
                    tag=f"train/loss",
                    scalar_value=loss.item(),
                    global_step=step,
                )
                print(f"step={step}, train_loss={round(loss.item(), 5)}")

            # Logging evaluation loss and visualization
            if step % self.cfg.trainer.eval_every_steps == 0:
                eval_loss, full_img_out, full_ground_truth = self.eval(full_coords=True)
                _, region_img_out, region_ground_truth = self.eval(full_coords=False)

                self._tb_writer.add_scalar(
                    tag="eval/loss",
                    scalar_value=eval_loss,
                    global_step=step,
                )

                # Log model output image on the full coordinates
                self._tb_writer.add_image(
                    "full/eval_out_full", full_img_out, global_step=step
                )
                self._tb_writer.add_image(
                    "full/gt_full", full_ground_truth, global_step=step
                )

                # Log model output image on the current region
                region_id = self.dataset.cur_region
                self._tb_writer.add_image(
                    f"region/eval_out_region{region_id}",
                    region_img_out,
                    global_step=step,
                )
                self._tb_writer.add_image(
                    f"region/gt_region{region_id}",
                    region_ground_truth,
                    global_step=step,
                )

                # Print eval stats
                print(f"step={step}, eval_loss={round(eval_loss, 5)}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            step += 1

    @torch.no_grad()
    def eval(self, full_coords: bool) -> Tuple[float, Tensor]:
        """Evaluate the current `self.siren` model.
        Args:
            full_coords: Whether to evaluate on the entire coordinate grid or the
                current coordinate region.
        Returns:
            eval_loss: Evaluation loss on the entire input image.
            img_out (side_length, side_length, 3): Model output as float RGB tensor for
                visualization.
            ground_truth (side_length, side_length, 3): Ground truth image for comparison.
        """
        self.siren.eval()

        if full_coords:
            model_input, ground_truth = (
                self.dataset.full_coords,
                self.dataset.full_pixels,
            )
        else:
            model_input, ground_truth = self.dataset.coords, self.dataset.pixels

        model_output = self.siren(model_input)[0]
        side_length = int(math.sqrt(model_output.shape[0]))

        eval_loss = ((model_output - ground_truth) ** 2).mean().item()

        # Recover spatial dimension for visualization
        img_out = (
            model_output.cpu()
            .view(side_length, side_length, -1)
            .detach()
            .permute(2, 0, 1)
        )

        # Clamp image in [0, 1] for visualization
        img_out = torch.clip(img_out, 0, 1)

        self.siren.train()

        return eval_loss, img_out, ground_truth
