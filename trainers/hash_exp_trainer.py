import logging

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn

import utils
from trainers.simple_trainer import SimpleTrainer

log = logging.getLogger(__name__)


class HashExpTrainer(SimpleTrainer):
    """Trainer to conduct experiments on HashNet, such as freezing the hash encoding
    loaded from a well-trained network and train the small MLP from scratch.
    """

    def __init__(self, cfg: DictConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        assert not (
            self.cfg.trainer.freeze_hash and cfg.trainer.freeze_mlp
        ), "Freezing both hash encoding and MLP. This will not train at all!"

        self.frozen_param_sum = 0  # For checking that frozen parameters are not updated

        #### Freeze part of the network and re-initialize the other ####
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
                    # Re-initialize hash encoding
                    nn.init.uniform_(module.weight, a=-1e-4, b=1e-4)
                    trainable_params.append(module.weight)

        # Re-initialize optimizer to contain only the trainable parameters
        self.optimizer = instantiate(
            self.cfg.trainer.optimizer, params=trainable_params
        )
        self.lr_scheduler = instantiate(
            self.cfg.trainer.lr_scheduler, optimizer=self.optimizer
        )

    def maybe_eval_and_log(self) -> None:
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
