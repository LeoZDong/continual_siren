import logging
import math
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from utils import mse2psnr

# A logger for this file
log = logging.getLogger(__name__)


class SimpleTrainer:
    """Simple continual learning trainer. Used as either the default trainer or the base
    class for other trainers.
    """

    def __init__(self, cfg: DictConfig, **kwargs):
        self.cfg = cfg
        self.continual = self.cfg.trainer.continual
        self.network = instantiate(cfg.network)
        self.dataset = instantiate(cfg.data)

        # Set device
        if not cfg.trainer.gpu_if_available:
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            # GPU acceleration for Apple silicon
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.network = self.network.to(self.device)

        self.optimizer = instantiate(
            self.cfg.trainer.optimizer, params=self.network.parameters()
        )
        self.lr_scheduler = instantiate(
            self.cfg.trainer.lr_scheduler, optimizer=self.optimizer
        )
        self.l1_lambda = self.cfg.trainer.l1_lambda

        self._work_dir = os.getcwd()
        self._tb_writer = SummaryWriter(self._work_dir)
        self._checkpoint_dir = os.path.join(os.getcwd(), "ckpt")
        os.mkdir(self._checkpoint_dir)

        self.step = 0

        # Load checkpoint
        if self.cfg.trainer.load_ckpt_path is not None:
            path = os.path.join(get_original_cwd(), self.cfg.trainer.load_ckpt_path)
            ckpt = torch.load(path)
            self.network.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            # Note that we still keep self.step at 0 and do not load it

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

        progress_bar = tqdm(total=self.cfg.trainer.total_steps)
        while self.step < self.cfg.trainer.total_steps:
            model_input, ground_truth = self.get_next_step_data(
                model_input, ground_truth
            )

            model_output, coords = self.network(model_input)
            loss = self.loss(model_output, ground_truth)
            self.maybe_log(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.step += 1
            progress_bar.update(1)

            # Evaluate
            self.maybe_eval_and_log()

            # Save checkpoint
            self.maybe_save_checkpoint(loss)

        progress_bar.close()

    @torch.no_grad()
    def eval(
        self,
        eval_coords: Union[str, int],
        output_img: bool = True,
    ) -> Tuple[Optional[float], Optional[Tensor], Optional[Tensor]]:
        """Evaluate the current `self.network` model.
        Args:
            eval_coords: If 'full', eval on the entire coordinate grid. If int, eval on
                the specified region index. If 'current', eval on the current region.
            output_img: Whether to return model output and ground truth as RGB image.

        Returns:
            eval_mse: Evaluation MSE loss on the specified `eval_coords`.
            img_out (side_length, side_length, 3): Model output for the specified
                `eval_coords` as RGB tensor.
            ground_truth (side_length, side_length, 3): Ground truth image for comparison.
        """
        self.network.eval()

        # Pick the specified coordinate grid to evaluate on
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

        model_output = self.network(model_input)[0]
        eval_mse = self.mse_loss(model_output, ground_truth).item()

        self.network.train()

        if not output_img:
            return eval_mse, None, None

        # Recover spatial dimension for visualization
        side_length = int(math.sqrt(model_output.shape[0]))
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

        return eval_mse, img_out, gt_img_out

    def mse_loss(self, model_output: Tensor, ground_truth: Tensor) -> Tensor:
        """Calculate MSE loss."""
        return ((model_output - ground_truth) ** 2).mean()

    def loss(self, model_output: Tensor, ground_truth: Tensor, **kwargs) -> Tensor:
        """Calculate overall loss, including any regularization loss."""
        loss = self.mse_loss(model_output, ground_truth)
        # L1 penalty for weight sparsity
        if self.l1_lambda != 0:
            l1_norm = sum(torch.norm(param, p=1) for param in self.network.parameters())
            loss += self.l1_lambda * l1_norm
        return loss

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
                model_input, ground_truth = next(self.dataloader_iter)
            except StopIteration:
                self.dataloader_iter = iter(self.dataloader)
                model_input, ground_truth = next(self.dataloader_iter)
            return model_input.to(self.device), ground_truth.to(self.device)

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
            log.info(
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
            log.info(f"step={self.step}, train_loss={round(loss.item(), 5)}")

    def maybe_eval_and_log(self) -> None:
        """Evaluate and log evaluation summary if appropriate."""
        if self.step % self.cfg.trainer.eval_every_steps == 0:
            # Evaluate on the entire image
            eval_mse_full, full_img_out, full_gt_img = self.eval(
                eval_coords="full", output_img=True
            )
            # Log evaluation loss for the full image
            psnr_full = mse2psnr(eval_mse_full)
            self._tb_writer.add_scalar(
                tag="eval/psnr_on_full_img",
                scalar_value=psnr_full,
                global_step=self.step,
            )

            # Evaluate on each region individually
            regions = np.arange(self.dataset.num_regions)
            eval_mse_backward = []
            for region in regions:
                eval_mse = self.eval(eval_coords=region, output_img=False)[0]
                if region < self.dataset.cur_region:
                    eval_mse_backward.append(eval_mse)

                # Log evaluation loss for each region
                psnr = mse2psnr(eval_mse)
                self._tb_writer.add_scalar(
                    tag=f"eval/psnr_on_region{region}",
                    scalar_value=psnr,
                    global_step=self.step,
                )

            # Log evaluation loss for backward regions
            if self.continual and len(eval_mse_backward) > 0:
                eval_mse_backward = np.mean(eval_mse_backward)
                psnr_backward = mse2psnr(eval_mse_backward)
                self._tb_writer.add_scalar(
                    tag="eval/psnr_on_backward_regions",
                    scalar_value=psnr_backward,
                    global_step=self.step,
                )

            # Record model output image on the full coordinate
            self._tb_writer.add_image(
                "full/eval_out_full", full_img_out, global_step=self.step
            )
            if self.step == 0:
                self._tb_writer.add_image(
                    "full/gt_full", full_gt_img, global_step=self.step
                )

            # Print eval stats
            log.info(f"step={self.step}, eval_psnr_full={round(psnr_full, 5)}")

        if self.is_final_step(self.step):
            eval_mse_full, full_img_out, full_gt_img = self.eval(
                eval_coords="full", output_img=True
            )

            # Save image output in final step
            torchvision.utils.save_image(
                full_img_out, os.path.join(self._work_dir, "full_img_out.png")
            )
            torchvision.utils.save_image(
                full_gt_img, os.path.join(self._work_dir, "full_gt_img.png")
            )

            # Record final performance in file
            f = open("final_result.txt", "w")
            f.write(f"psnr_full={mse2psnr(eval_mse_full)}\n")
            regions_write = ""
            for region in np.arange(self.dataset.num_regions):
                eval_mse = self.eval(eval_coords=region, output_img=False)[0]
                regions_write += f"{mse2psnr(eval_mse)}\t"
            f.write("psnr_regions:\n")
            f.write(regions_write)
            f.close()

    def maybe_save_checkpoint(self, loss: Tensor) -> None:
        if self.is_final_step(self.step):
            name = f"final"
        elif self.step % self.cfg.trainer.checkpoint_every_steps == 0:
            name = f"ckpt_{self.step}"
        else:
            return

        log.info(f"Saving checkpoint at step={self.step}...")
        torch.save(
            {
                "step": self.step,
                "model_state_dict": self.network.state_dict(),
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
