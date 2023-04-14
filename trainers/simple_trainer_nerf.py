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
        self.num_steps_per_ray = self.network.num_steps_per_ray

    def train(self) -> None:
        if self.network.optimized_march_cuda:
            # TODO: Technically unnecessary to put in eval mode. Remove in the future.
            self.network.eval()
            self.network.mark_invisible_cells(
                move_to(self.dataset.K, self.device),
                move_to(self.dataset.poses, self.device),
                (self.dataset.img_w, self.dataset.img_h),
                chunk=64**3,
            )
            self.network.train()

        progress_bar = tqdm(total=self.total_steps)
        while self.step < self.total_steps:
            # Extra book-keeping for optimized CUDA ray marching
            self.maybe_update_density_bitfield()

            model_input, ground_truth = self.get_next_step_data()

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
    def maybe_update_density_bitfield(self) -> None:
        if (
            self.network.optimized_march_cuda
            and self.step % self.update_density_bitfield_every_steps == 0
        ):
            self.network.eval()
            self.network.update_density_grid(
                density_threshold=0.01 * self.num_steps_per_ray / 3**0.5,
                warmup=self.step < self.warmup_steps,
            )
            self.network.train()

    @torch.no_grad()
    def eval(
        self,
        eval_coords: Union[str, int],
        output_img: bool,
        max_eval_size: int = 6400,
    ) -> Tuple[Optional[float], Optional[List[Tensor]], Optional[List[Tensor]]]:
        """Evaluate the current `self.network` model. See super class for more documentation.
        Args:
            max_eval_size: Maximum number of samples to evaluate on. If evaluation set
                exceeds this size, randomly sample `max_eval_size` to evaluate on.
        """
        self.network.eval()

        # Pick the specified coordinate grid to evaluate on
        if eval_coords == "current":
            model_input = self.dataset.input
            ground_truth = self.dataset.output
        elif eval_coords == "full":
            model_input = self.dataset.full_input
            ground_truth = self.dataset.full_output
        else:
            model_input = self.dataset.input_regions[eval_coords]
            ground_truth = self.dataset.output_regions[eval_coords]

        # Subsample if evaluating too many points
        if ground_truth.shape[0] > max_eval_size:
            perm = torch.randperm(ground_truth.shape[0])[:max_eval_size]
            input = {}
            for key, val in model_input.items():
                input[key] = val[perm]
            ground_truth = ground_truth[perm]
        else:
            input = model_input

        input = move_to(input, self.device)
        ground_truth = move_to(ground_truth, self.device)

        model_output = self.network(**input, use_test_render=False)
        eval_mse = self.mse_loss(model_output, ground_truth).item()

        if not output_img:
            self.network.train()
            return eval_mse, None, None

        # We pick the first frame of each region to output
        # TODO: This might be too slow if there are many regions?
        # TODO: This is not the most elegant way to retrieve wh dimensions...
        imgs_out = []
        gts_img_out = []
        num_pixels_per_img = self.dataset.img_w * self.dataset.img_h
        print("Producing output image for each region...")
        for region in tqdm(range(self.dataset.num_regions)):
            model_input = {}
            for key, val in self.dataset.input_regions[region].items():
                model_input[key] = val[:num_pixels_per_img].to(self.device)
            model_output = self.network(
                **model_input, use_test_render=False, return_cpu=True
            )
            ground_truth = self.dataset.output_regions[region][:num_pixels_per_img]
            imgs_out.append(
                model_output.cpu()
                .view(self.dataset.img_w, self.dataset.img_h, -1)
                .detach()
                .permute(2, 0, 1)
                .clip(0, 1)
            )
            gts_img_out.append(
                ground_truth.cpu()
                .view(self.dataset.img_w, self.dataset.img_h, -1)
                .detach()
                .permute(2, 0, 1)
            )
            torch.cuda.empty_cache()

        self.network.train()
        return eval_mse, imgs_out, gts_img_out

    @torch.no_grad()
    def eval_for_video(self) -> List[np.ndarray]:
        """Evaluate NeRF on a set of pre-generated "video poses" to output the
        test video frames.
        """
        self.network.eval()

        img_h = self.dataset.img_h
        img_w = self.dataset.img_w
        video_frames = []

        # If ground truth is available, also evaluate psnr
        if self.dataset.video_outputs is not None:
            psnr_regions = [[] for _ in range(self.dataset.num_regions)]
            psnrs = []

        print(f"Evaluating for {len(self.dataset.video_inputs)} test frames...")
        for i in tqdm(range(len(self.dataset.video_inputs))):
            model_input = move_to(self.dataset.video_inputs[i], self.device)
            model_output = self.network(
                **model_input, use_test_render=False, return_cpu=True
            )
            # Recover width and height dimensions (num_rays, 3) -> (w, h, 3)
            img_out = model_output.cpu().view(img_w, img_h, -1).detach().numpy()
            # Convert to uint
            img_out = (img_out * 255).astype(np.uint8)
            video_frames.append(img_out)

            # If ground truth is available, also evaluate psnr
            if self.dataset.video_outputs is not None:
                region = self.dataset.region_simulator.get_region_idx(
                    model_input["rays_o"][0]
                )
                ground_truth = self.dataset.video_outputs[i]
                mse = self.mse_loss(model_output, ground_truth).item()
                psnr = mse2psnr(mse)
                psnrs.append(psnr)
                psnr_regions[region].append(psnr)

        self.network.train()

        if self.dataset.video_outputs is not None:
            psnr_full = sum(psnrs) / len(psnrs)
            psnr_regions = [
                sum(psnr_region) / len(psnr_region) for psnr_region in psnr_regions
            ]
            return video_frames, psnr_full, psnr_regions
        else:
            return video_frames, None, None

    def maybe_eval_and_log(self) -> None:
        """Evaluate and log evaluation summary if appropriate."""
        if self.step % self.cfg.trainer.eval_every_steps == 0:
            ## Full region ##
            # Evaluate on all regions
            eval_mse_full, imgs_out, gt_imgs_out = self.eval(
                eval_coords="full", output_img=True
            )
            # Log evaluation loss for all regions to tensorboard
            psnr_full = mse2psnr(eval_mse_full)
            self._tb_writer.add_scalar(
                tag="eval/psnr_on_full_img",
                scalar_value=psnr_full,
                global_step=self.step,
            )
            # Record model output image on all regions (1 frame per region)
            for i, (img, gt_img) in enumerate(zip(imgs_out, gt_imgs_out)):
                self._tb_writer.add_image(
                    f"full/eval_out_{i}", img, global_step=self.step
                )

                if self.step == self.cfg.trainer.eval_every_steps:
                    self._tb_writer.add_image(
                        f"full/gt_{i}", gt_img, global_step=self.step
                    )

            # Log eval stats to file
            log.info(f"step={self.step}, eval_psnr_full={round(psnr_full, 5)}")

            ## Individual regions ##
            # Evaluate on each region individually
            regions = np.arange(self.dataset.num_regions)
            eval_mse_backward = []
            for region in regions:
                eval_mse = self.eval(eval_coords=region, output_img=False)[0]
                if region < self.dataset.cur_region:
                    eval_mse_backward.append(eval_mse)

                # Log eval loss for each region
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

            ## Final records ##
            if self.is_final_step(self.step):
                # Generate test video
                video_frames, psnr_full, psnr_regions = self.eval_for_video()

                # Write the images to the gif file
                with imageio.get_writer(
                    os.path.join(self._work_dir, f"full_scene_out.gif"),
                    mode="I",
                    fps=30.0,
                ) as writer:
                    for frame in video_frames:
                        writer.append_data(frame)

                # Record final performance in file
                if psnr_full is not None:
                    f = open("final_result.txt", "w")
                    f.write(f"psnr_full={psnr_full}\n")
                    regions_write = ""
                    for psnr_region in psnr_regions:
                        regions_write += f"{psnr_region}\t"

                    f.write("psnr_regions:\n")
                    f.write(regions_write)
                    f.close()

            torch.cuda.empty_cache()
