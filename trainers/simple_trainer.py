import os

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

import utils


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
        model_input, ground_truth = next(iter(self.dataset))
        model_input, ground_truth = model_input.to(self.device), ground_truth.to(
            self.device
        )

        for step in range(total_steps + 1):
            model_output, coords = self.siren(model_input)
            loss = ((model_output - ground_truth) ** 2).mean()

            # Logging
            if step % self.cfg.trainer.summary_every_steps == 0:
                self._tb_writer.add_scalar(
                    tag=f"train/loss",
                    scalar_value=loss.item(),
                    global_step=step,
                )
                img_out = (
                    model_output.cpu().view(256, 256, -1).detach().permute(2, 0, 1)
                )
                self._tb_writer.add_image("train", img_out, global_step=step)
                print(f"step={step}, loss={round(loss.item(), 5)}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
