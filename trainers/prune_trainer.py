import logging

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from trainers.simple_trainer import SimpleTrainer
from utils import prune_model

log = logging.getLogger(__name__)


class PruneTrainer(SimpleTrainer):
    """Trainer that implements intermittent pruning during training."""

    def __init__(self, cfg: DictConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.prune_amount = cfg.trainer.prune_amount

    def train(self) -> None:
        # Prepare data:
        # Only used in continual setting: return all points in the current region.
        # Each "batch" is all points in the current region.
        model_input, ground_truth = self.dataset.coords, self.dataset.pixels
        model_input, ground_truth = model_input.to(self.device), ground_truth.to(
            self.device
        )

        progress_bar = tqdm(total=self.cfg.trainer.total_steps)
        self.step = 0
        while self.step < self.cfg.trainer.total_steps:
            model_input, ground_truth = self.get_next_step_data(
                model_input, ground_truth
            )

            #### Model pruning phase ####
            self.maybe_prune()

            #### Signal fitting phase ####
            model_output, coords = self.siren(model_input)
            loss = self.loss(model_output, ground_truth)

            # L1 penalty for weight sparsity
            if self.l1_lambda != 0:
                l1_norm = sum(
                    torch.norm(param, p=1) for param in self.siren.parameters()
                )
                loss += self.l1_lambda * l1_norm

            self.maybe_log(loss)
            self.maybe_eval_and_log()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.step += 1
            progress_bar.update(1)

            # Save checkpoint
            self.maybe_save_checkpoint(loss)

        progress_bar.close()

    def maybe_prune(self) -> None:
        if self.step % self.cfg.trainer.prune_every_steps == 0:
            log.info(f"Pruning at step {self.step}...")
            prune_model(self.siren, self.prune_amount, finalize_pruning=True)
