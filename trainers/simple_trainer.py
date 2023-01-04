import math
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.utils import data

# from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter

from utils import mse2psnr


class SimpleTrainer:
    def __init__(self, cfg: DictConfig, **kwargs):
        self.cfg = cfg
        self.continual = self.cfg.trainer.continual
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
        self.l1_lambda = self.cfg.trainer.l1_lambda

        self._work_dir = os.getcwd()
        self._tb_writer = SummaryWriter(self._work_dir)
        self._checkpoint_dir = os.path.join(os.getcwd(), "ckpt")
        os.mkdir(self._checkpoint_dir)

        # Only used in non-continual setting: randomly sample points from the full grid
        # Each "batch" has the same number of points as *one* region in the continual case.
        self.dataloader = data.DataLoader(
            self.dataset,
            batch_size=len(self.dataset) // self.dataset.num_regions,
            shuffle=True,
        )
        self.dataloader_iter = iter(self.dataloader)

    def train(self) -> None:
        # Prepare data:
        # Only used in continual setting: return all points in the current region.
        # Each "batch" is all points in the current region.
        model_input, ground_truth = self.dataset.coords, self.dataset.pixels
        model_input, ground_truth = model_input.to(self.device), ground_truth.to(
            self.device
        )

        self.step = 0
        while self.step < self.cfg.trainer.total_steps:
            model_input, ground_truth = self.get_next_step_data(
                model_input, ground_truth
            )

            model_output, coords = self.siren(model_input)
            loss = ((model_output - ground_truth) ** 2).mean()
            self.maybe_log(loss)

            # L1 penalty for weight sparsity
            if self.l1_lambda != 0:
                l1_norm = sum(
                    torch.norm(param, p=1) for param in self.siren.parameters()
                )
                loss += self.l1_lambda * l1_norm

            self.maybe_eval_and_log()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.step += 1

            # Save checkpoint
            self.maybe_save_checkpoint(loss)

    @torch.no_grad()
    def eval(
        self, eval_coords: Union[str, int]
    ) -> Tuple[Optional[float], Optional[Tensor], Optional[Tensor]]:
        """Evaluate the current `self.siren` model.
        Args:
            eval_coords: If 'full', eval on the entire coordinate grid. If int, eval on
                the specified region index. If 'current', eval on the current region. If
                'backward', eval on all regions before and not including current region.

            Whether to evaluate on the entire coordinate grid or evaluate
                separately on each coordinate region.

        Returns:
            eval_losses: Evaluation MSE loss on the specified `eval_coords`.
            img_out (side_length, side_length, 3): Model output for the specified
                `eval_coords` as RGB tensor. If `eval_coords == 'backward'`, return None.
            ground_truth (side_length, side_length, 3): Ground truth image for comparison.
                If `eval_coords == 'backward'`, return None.
        """
        self.siren.eval()

        #### Evaluate on multiple coordinate grids. No output image. ####
        if eval_coords == "backward":
            regions = np.arange(self.dataset.num_regions)
            backward_idxs = regions[regions < self.dataset.cur_region]

            # If current region is 0, then there is no backward loss to be calculated
            if len(backward_idxs) == 0:
                return 1, None, None

            eval_loss = 0
            for backward_idx in backward_idxs:
                model_input, ground_truth = (
                    self.dataset.coords_regions[backward_idx].to(self.device),
                    self.dataset.pixels_regions[backward_idx].to(self.device),
                )
                model_output = self.siren(model_input)[0]
                eval_loss += ((model_output - ground_truth) ** 2).mean().item()

            return eval_loss, None, None

        #### Evaluate on specified coordinate grid ####
        if eval_coords == "current":
            model_input, ground_truth = (
                self.dataset.coords.to(self.device),
                self.dataset.pixels.to(self.device),
            )
        elif eval_coords == "full":
            model_input, ground_truth = (
                self.dataset.full_coords.to(self.device),
                self.dataset.full_pixels.to(self.device),
            )
        else:
            model_input, ground_truth = (
                self.dataset.coords_regions[eval_coords].to(self.device),
                self.dataset.pixels_regions[eval_coords].to(self.device),
            )

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
        gt_img_out = (
            ground_truth.cpu()
            .view(side_length, side_length, -1)
            .detach()
            .permute(2, 0, 1)
        )

        # Clamp image in [0, 1] for visualization
        img_out = torch.clip(img_out, 0, 1)

        self.siren.train()

        return eval_loss, img_out, gt_img_out

    def get_next_step_data(
        self, model_input: Tensor, ground_truth: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Get data for the next training step.
        In the continual case, data for the next step does not change unless we need
            to switch to a differen region.
        In the non-continual case, data for the next step is the next batch of randomly
            sampled points in the full grid.
        """
        if self.continual:
            return self.maybe_switch_region(model_input, ground_truth)
        else:
            # Look out for stop iteration
            try:
                return next(self.dataloader_iter)
            except StopIteration:
                self.dataloader_iter = iter(self.dataloader)
                return next(self.dataloader_iter)

    def maybe_switch_region(
        self, model_input: Tensor, ground_truth: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Switch to new region for training if appropriate.
        Args:
            model_input, ground_truth: Current feature-label pair.
        Returns:
            If time to switch, return the new feature-label pair. Else, return the
                current feature-label pair.
        """
        if self.need_to_switch_region:
            new_region = (self.dataset.cur_region + 1) % self.dataset.num_regions
            print(
                f"step={self.step}. Switching region from {self.dataset.cur_region} to {new_region}!"
            )
            self.dataset.set_cur_region(new_region)
            model_input, ground_truth = self.dataset.coords, self.dataset.pixels
            model_input, ground_truth = model_input.to(self.device), ground_truth.to(
                self.device
            )
            return model_input, ground_truth
        else:
            return model_input, ground_truth

    def maybe_log(self, loss: Tensor) -> None:
        """Log training summary if appropriate."""
        if self.step % self.cfg.trainer.summary_every_steps == 0:
            self._tb_writer.add_scalar(
                tag=f"train/loss_on_cur_region",
                scalar_value=loss.item(),
                global_step=self.step,
            )
            psnr = mse2psnr(loss.item())
            self._tb_writer.add_scalar(
                tag=f"train/psnr_on_cur_region",
                scalar_value=psnr,
                global_step=self.step,
            )

            print(f"step={self.step}, train_loss={round(loss.item(), 5)}")

    def maybe_eval_and_log(self) -> None:
        """Evaluate and log evaluation summary if appropriate."""
        if self.step % self.cfg.trainer.eval_every_steps == 0:
            eval_loss_full, full_img_out, full_gt_img = self.eval(eval_coords="full")
            eval_loss_backward, _, _ = self.eval(eval_coords="backward")
            _, region_img_out, region_gt_img = self.eval(eval_coords="current")

            # Log evaluation losses
            self._tb_writer.add_scalar(
                tag="eval/loss_on_full_img",
                scalar_value=eval_loss_full,
                global_step=self.step,
            )
            psnr_full = mse2psnr(eval_loss_full)
            self._tb_writer.add_scalar(
                tag="eval/psnr_on_full_img",
                scalar_value=psnr_full,
                global_step=self.step,
            )

            if self.continual:
                self._tb_writer.add_scalar(
                    tag="eval/loss_on_backward_regions",
                    scalar_value=eval_loss_backward,
                    global_step=self.step,
                )
                psnr_backward = mse2psnr(eval_loss_backward)
                self._tb_writer.add_scalar(
                    tag="eval/psnr_on_backward_regions",
                    scalar_value=psnr_backward,
                    global_step=self.step,
                )

            # Log model output image on the full coordinates
            self._tb_writer.add_image(
                "full/eval_out_full", full_img_out, global_step=self.step
            )
            if self.step == 0:
                self._tb_writer.add_image(
                    "full/gt_full", full_gt_img, global_step=self.step
                )

            if self.continual:
                # Log model output image on the current region
                region_id = self.dataset.cur_region
                self._tb_writer.add_image(
                    f"region/eval_out_region{region_id}",
                    region_img_out,
                    global_step=self.step,
                )

                # TODO: Only log ground-truth for comparison once (because it stays the same)
                self._tb_writer.add_image(
                    f"region/gt_region{region_id}",
                    region_gt_img,
                    global_step=self.step,
                )

            # Print eval stats
            print(
                f"step={self.step}, eval_loss_full={round(eval_loss_full, 5)}, eval_loss_backward={round(eval_loss_backward, 5)}"
            )

        if self.is_final_step(self.step + 1):
            # Save image output in final step
            eval_loss, full_img_out, full_gt_img = self.eval(eval_coords="full")
            torchvision.utils.save_image(
                full_img_out, os.path.join(self._work_dir, "full_img_out.png")
            )
            torchvision.utils.save_image(
                full_gt_img, os.path.join(self._work_dir, "full_gt_img.png")
            )

    def maybe_save_checkpoint(self, loss: Tensor) -> None:
        if self.is_final_step(self.step):
            name = f"final"
        elif self.step % self.cfg.trainer.checkpoint_every_steps == 0:
            name = f"ckpt_{self.step}"
        else:
            return

        print(f"Saving checkpoint at step={self.step}...")
        torch.save(
            {
                "step": self.step,
                "model_state_dict": self.siren.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss.item(),
            },
            os.path.join(self._checkpoint_dir, f"{name}.pt"),
        )

    def is_final_step(self, step) -> bool:
        return step == self.cfg.trainer.total_steps

    @property
    def need_to_switch_region(self) -> bool:
        return self.step and self.step % self.cfg.trainer.switch_region_every_steps == 0
