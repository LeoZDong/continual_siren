import logging
from typing import Tuple

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor, nn

import utils
from trainers.simple_trainer import SimpleTrainer

log = logging.getLogger(__name__)


class HashExpTrainer(SimpleTrainer):
    """Trainer to conduct experiments on HashNet, such as freezing the hash encoding
    loaded from a well-trained network and train the small MLP from scratch.
    """

    def __init__(self, cfg: DictConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.freeze_hash = cfg.trainer.freeze_hash
        self.freeze_mlp = cfg.trainer.freeze_mlp
        self.freeze_at = cfg.trainer.freeze_at

        assert not (
            self.cfg.trainer.freeze_hash and cfg.trainer.freeze_mlp
        ), "Freezing both hash encoding and MLP. This will not train at all!"
        assert self.freeze_at in ["init", "end_of_region0"]

        if self.freeze_at == "init":
            self.freeze_part_of_network(reinit_unfrozen_part=True)

    def freeze_part_of_network(self, reinit_unfrozen_part: bool) -> None:
        """Freeze part of the network (either hash encoding or MLP), specified by the
        config. Optionally re-initialize the unfrozen part. This can be called either:
            1) At initialization. We typically expect the network to load a checkpoint
               from a perfectly-trained network. We then freeze the "perfect" values of
               one part, and re-initialize & train the other part **from scratch**.
            2) During training. We typically would let the entire network to train for
               some time (e.g. for one region). Then we freeze one part and **continue**
               training the other part.
        Args:
            reinit_unfrozen_part: Whether to re-initialize the unfrozen part of the
                network to train from scratch.
        """
        log.info(
            f"Freezing {'hash' if self.freeze_hash else 'MLP'} of the network at step {self.step}...\n"
            f"{'Re-initialize' if reinit_unfrozen_part else 'Do not re-initialize'} {'hash' if not self.freeze_hash else 'MLP'}!"
        )

        self.frozen_param_sum = 0  # For checking that frozen parameters are not updated
        module_names = utils.get_module_names(self.network)
        named_modules_dict = dict(self.network.named_modules())

        trainable_params = []

        for name in module_names:
            module = named_modules_dict[name]
            if self.cfg.trainer.freeze_hash:
                if name.startswith("hash_embedding"):
                    # Freeze hash encoding
                    module.weight.requires_grad = False
                    self.frozen_param_sum += module.weight.sum().item()
                else:
                    if reinit_unfrozen_part:
                        # Re-initialize MLP
                        module.reset_parameters()
                        nn.init.xavier_uniform_(module.weight)
                    trainable_params.append(module.weight)
                    trainable_params.append(module.bias)
            elif self.cfg.trainer.freeze_mlp:
                if name.startswith("net"):
                    # Freeze MLP
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False
                    self.frozen_param_sum += module.weight.sum().item()
                    self.frozen_param_sum += module.bias.sum().item()
                else:
                    if reinit_unfrozen_part:
                        # Re-initialize hash encoding
                        nn.init.uniform_(module.weight, a=-1e-4, b=1e-4)
                    trainable_params.append(module.weight)

        # Reset optimizer to contain only the trainable parameters
        self.optimizer = instantiate(
            self.cfg.trainer.optimizer, params=trainable_params
        )
        self.lr_scheduler = instantiate(
            self.cfg.trainer.lr_scheduler, optimizer=self.optimizer
        )

    def maybe_switch_region(
        self, model_input: Tensor, ground_truth: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Overwrite to freeze part of the network when appropriate."""
        if self.freeze_at == "end_of_region0" and (
            self.dataset.cur_region == 0 and self.need_to_switch_region
        ):
            self.freeze_part_of_network(reinit_unfrozen_part=False)

        return super().maybe_switch_region(model_input, ground_truth)

    def maybe_eval_and_log(self) -> None:
        """Overwrite to check that frozen parameters are not changed at last step."""
        super().maybe_eval_and_log()

        # Check that frozen parameters are never updated
        # Probably remove in the future, but a cheap verification for now
        if self.is_final_step(self.step + 1):
            frozen_param_sum = 0
            for name, param in self.network.named_parameters():
                if self.cfg.trainer.freeze_hash and name.startswith("hash_embedding"):
                    frozen_param_sum += param.sum().item()
                elif self.cfg.trainer.freeze_mlp and name.startswith("net"):
                    frozen_param_sum += param.sum().item()
            try:
                assert (
                    frozen_param_sum == self.frozen_param_sum
                ), f"Frozen parameters before and after training have changed! Before: {self.frozen_param_sum}, after: {frozen_param_sum}"
            except AssertionError as e:
                log.error(f"Assertion error: {str(e)}")
                raise
