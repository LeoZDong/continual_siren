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
    """Same training procedure as SimpleTrainer, but different evaluation and testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @torch.no_grad()
    def eval(
        self,
        eval_coords: Union[str, int],
        output_img: bool = True,
        max_eval_size: int = 65536,
    ) -> Tuple[Optional[float], Optional[Tensor], Optional[Tensor]]:
        """Evaluate the current `self.network` model."""
        assert not output_img

        self.network.eval()

        t = time.time()
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

        print(f"Prepare time: {time.time() - t}")

        # Subsample if evaluating too many points
        if ground_truth.shape[0] > max_eval_size:
            # print(f"Subsampling to {max_eval_size}")
            t = time.time()
            perm = torch.randperm(ground_truth.shape[0])[:max_eval_size]
            input = {}
            for key, val in model_input.items():
                input[key] = val[perm]
            ground_truth = ground_truth[perm]
            # If we subsample, we can no longer return images
            output_img = False
            print(f"Subsample time: {time.time() - t}")
        else:
            input = model_input

        t = time.time()
        model_output = self.network(**input)
        print(f"Forward time: {time.time() - t}")

        t = time.time()
        eval_mse = self.mse_loss(model_output, ground_truth).item()
        print(f"Loss time: {time.time() - t}")

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
            model_output = self.network(**model_input)
            # Recover height and width dimensions (num_rays, 3) -> (h, w, 3)
            img_out = model_output.cpu().view(img_h, img_w, -1).detach().numpy()
            video_frames.append(img_out)

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

            # if self.is_final_step(self.step):
            # Generate test video
            video_frames = self.eval_for_video()

            # Write the images to the gif file
            with imageio.get_writer(
                os.path.join(self._work_dir, "full_scene_out.gif"),
                mode="I",
                fps=30.0,
            ) as writer:
                for frame in video_frames:
                    writer.append_data(frame)
