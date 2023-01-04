import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from trainers.simple_trainer import SimpleTrainer


class NodeSharpenTrainer(SimpleTrainer):
    def __init__(self, cfg: DictConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.sharpen_ratio = cfg.trainer.sharpen_ratio
        self.sharpen_factor = cfg.trainer.sharpen_factor
        self.sharpen_optimizer = instantiate(
            self.cfg.trainer.sharpen_optimizer, params=self.siren.parameters()
        )

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

            #### Node sharpening phase ####
            activations = self.siren.forward_with_activations(
                model_input, retain_grad=False
            )
            # Output layer has tag `len(layers_to_sharpen) - 2` because we exclude `input`
            # (first element in `layers_to_sharpen`) and it is zero-indexed.
            output_layer_name = (
                f"<class 'torch.nn.modules.linear.Linear'>_{len(activations) - 2}"
            )
            del activations[output_layer_name]
            del activations["input"]

            layers_to_sharpen = []
            for i, layer in enumerate(activations.values()):
                # Only sharpen last layer
                if i != len(activations) - 1:
                    continue
                layers_to_sharpen.append(layer)

            node_sharpening_loss = 0
            for layer in layers_to_sharpen:
                if self.sharpen_ratio == "log":
                    num_to_sharpen = int(np.log(layer.shape[1]))
                else:
                    num_to_sharpen = int(self.sharpen_ratio * layer.shape[1])

                # Indices of nodes sorted by activation magnitude (sorted once per input)
                nodes_sorted = torch.argsort(layer.abs(), dim=1, descending=True)
                idx_sharpen = nodes_sorted[:, :num_to_sharpen]
                idx_blunt = nodes_sorted[:, num_to_sharpen:]

                # We want the magnitude of `idx_sharpen` to be close to 1
                sharpen_loss = (
                    self.sharpen_factor
                    * (1 - torch.gather(layer, 1, idx_sharpen).abs()).sum()
                )
                # We want the magnitude of `idx_blunt` to be close to 0
                blunt_loss = (
                    self.sharpen_factor * torch.gather(layer, 1, idx_blunt).abs()
                ).sum()

                node_sharpening_loss += sharpen_loss + blunt_loss

            self.sharpen_optimizer.zero_grad()
            node_sharpening_loss.backward()
            self.sharpen_optimizer.step()

            #### Signal fitting phase ####
            model_output, coords = self.siren(model_input)
            loss = ((model_output - ground_truth) ** 2).mean()

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

            # Save checkpoint
            self.maybe_save_checkpoint(loss)
