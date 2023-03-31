import logging
import os
import time
from typing import List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from trainers.simple_trainer import SimpleTrainer
from utils import move_to, mse2psnr

# A logger for this file
log = logging.getLogger(__name__)


class SimpleTrainerNerf(SimpleTrainer):
    """Almost the same training procedure as SimpleTrainer except extra book-keeping
    for optimized CUDA ray marching. Different evaluation and testing.
    """

    def __init__(
        self, update_density_bitfield_every_steps: int, warmup_steps: int, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.update_density_bitfield_every_steps = update_density_bitfield_every_steps
        self.warmup_steps = warmup_steps

    def train(self) -> None:
        if self.network.optimized_march_cuda:
            self.network.mark_invisible_cells(
                move_to(self.dataset.K, self.device),
                move_to(self.dataset.poses, self.device),
                (self.dataset.img_w, self.dataset.img_h),
                chunk=64**3,
            )

        # Prepare data:
        # Only used in continual setting: return all points in the current region.
        # Each "batch" is all points in the current region.
        model_input, ground_truth = self.dataset.input, self.dataset.output
        model_input, ground_truth = move_to(model_input, self.device), move_to(
            ground_truth, self.device
        )

        progress_bar = tqdm(total=self.total_steps)
        while self.step < self.total_steps:
            # Extra book-keeping for optimized CUDA ray marching
            self.maybe_update_density_bitfield()

            model_input, ground_truth = self.get_next_step_data(
                model_input, ground_truth
            )

            model_output = self.network(**model_input)
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
    def maybe_update_density_bitfield(self):
        if (
            self.network.optimized_march_cuda
            and self.step % self.update_density_bitfield_every_steps == 0
        ):
            # TODO: Configure?
            MAX_SAMPLES = 1024
            self.network.update_density_grid(
                density_threshold=0.01 * MAX_SAMPLES / 3**0.5,
                warmup=self.step < self.warmup_steps,
            )

    @torch.no_grad()
    def eval(
        self,
        eval_coords: Union[str, int],
        output_img: bool = True,
        max_eval_size: int = 6400,
    ) -> Tuple[Optional[float], Optional[Tensor], Optional[Tensor]]:
        """Evaluate the current `self.network` model."""
        assert not output_img, "We cannot output image in NeRF trainer evaluation!"

        self.network.eval()

        # Pick the specified coordinate grid to evaluate on
        if eval_coords == "current":
            model_input, ground_truth = (
                move_to(self.dataset.input, self.device),
                move_to(self.dataset.output, self.device),
            )
        elif eval_coords == "full":
            model_input, ground_truth = (
                move_to(self.dataset.full_input, self.device),
                move_to(self.dataset.full_output, self.device),
            )
        else:
            model_input, ground_truth = (
                move_to(self.dataset.input_regions[eval_coords], self.device),
                move_to(self.dataset.output_regions[eval_coords], self.device),
            )

        # Subsample if evaluating too many points
        if ground_truth.shape[0] > max_eval_size:
            t = time.time()
            perm = torch.randperm(ground_truth.shape[0])[:max_eval_size]
            input = {}
            for key, val in model_input.items():
                input[key] = val[perm]
            ground_truth = ground_truth[perm]
        else:
            input = model_input

        # We can increase `max_ray_batch` in evaluation
        model_output = self.network(**input, max_ray_batch=4096)
        eval_mse = self.mse_loss(model_output, ground_truth).item()

        self.network.train()

        return eval_mse, None, None

    @torch.no_grad()
    def eval_for_video(self) -> List[np.ndarray]:
        """Evaluate NeRF on a set of pre-generated "video poses" to output the
        test video frames.
        """
        self.network.eval()

        img_h = self.dataset.img_h
        img_w = self.dataset.img_w
        video_frames = []
        print(f"Evaluating for {len(self.dataset.video_inputs)} test frames...")
        for model_input in tqdm(self.dataset.video_inputs):
            model_input = move_to(model_input, self.device)
            # We can increase `max_ray_batch` in evaluation
            model_output = self.network(**model_input, max_ray_batch=4096)
            # Recover height and width dimensions (num_rays, 3) -> (h, w, 3)
            img_out = model_output.cpu().view(img_h, img_w, -1).detach().numpy()
            # Convert to uint
            img_out = (img_out * 255).astype(np.uint8)
            video_frames.append(img_out)
            #### TEMP: Only Eval for first image! ####
            break

        self.network.train()
        return video_frames

    def maybe_eval_and_log(self) -> None:
        """Evaluate and log evaluation summary if appropriate."""
        if self.step % self.cfg.trainer.eval_every_steps == 0:
            # Evaluate on the entire image
            eval_mse_full, _, _ = self.eval(eval_coords="full", output_img=False)
            # Log evaluation loss for the full image to tensorboard
            psnr_full = mse2psnr(eval_mse_full)
            self._tb_writer.add_scalar(
                tag="eval/psnr_on_full_img",
                scalar_value=psnr_full,
                global_step=self.step,
            )

            # Log eval stats to file
            log.info(f"step={self.step}, eval_psnr_full={round(psnr_full, 5)}")

            # Generate test video
            video_frames = self.eval_for_video()

            # Write the images to the gif file
            with imageio.get_writer(
                os.path.join(self._work_dir, f"full_scene_out_{self.step}.gif"),
                mode="I",
                fps=30.0,
            ) as writer:
                for frame in video_frames:
                    writer.append_data(frame)
