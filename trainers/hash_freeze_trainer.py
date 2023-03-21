import logging
from typing import List, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor, nn

import utils
from trainers.simple_trainer import SimpleTrainer, SimpleTrainerGiga
from utils import move_to

log = logging.getLogger(__name__)


class HashFreezeTrainer(SimpleTrainer):
    """Trainer for HashNet that freezes one part (MLP or hash encoding) and re-intialize
    the other to train from scratch.

    This is useful for conducting experiments such as loading and freezing a perfectly
    trained MLP and training the hash encoding.

    Or it is simply good for mitigating forgetting in the MLP by freezing it, since most
    of the training happens in the hash encoding.
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

    def freeze_part_of_network(
        self,
        reinit_unfrozen_part: bool,
    ) -> None:
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
        module_names = utils.get_module_names(self.network_to_freeze)
        named_modules_dict = dict(self.network_to_freeze.named_modules())

        freeze_params = []
        for name in module_names:
            module = named_modules_dict[name]
            if self.cfg.trainer.freeze_hash:
                if isinstance(module, nn.Embedding):
                    # Freeze hash encoding
                    module.weight.requires_grad = False
                    self.frozen_param_sum += module.weight.sum().item()
                    freeze_params.append(module.weight)
                else:
                    if reinit_unfrozen_part:
                        # Re-initialize MLP
                        module.reset_parameters()
                        nn.init.xavier_uniform_(module.weight)
            elif self.cfg.trainer.freeze_mlp:
                if isinstance(module, nn.Linear):
                    # Freeze MLP
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False
                    self.frozen_param_sum += module.weight.sum().item()
                    self.frozen_param_sum += module.bias.sum().item()
                    freeze_params.append(module.weight)
                    freeze_params.append(module.bias)
                else:
                    if reinit_unfrozen_part:
                        # Re-initialize hash encoding
                        nn.init.uniform_(module.weight, a=-1e-4, b=1e-4)

        # Reset optimizer to remove frozen parameters
        current_params = [
            param for group in self.optimizer.param_groups for param in group["params"]
        ]
        current_params = list(set(current_params) - set(freeze_params))
        self.optimizer = instantiate(self.cfg.trainer.optimizer, params=current_params)
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

            module_names = utils.get_module_names(self.network)
            named_modules_dict = dict(self.network.named_modules())
            for name in module_names:
                module = named_modules_dict[name]
                if self.cfg.trainer.freeze_hash and isinstance(module, nn.Embedding):
                    # Freeze hash encoding
                    frozen_param_sum += module.weight.sum().item()
                elif self.cfg.trainer.freeze_mlp and isinstance(module, nn.Linear):
                    # Freeze MLP
                    frozen_param_sum += module.weight.sum().item()
                    frozen_param_sum += module.bias.sum().item()

            try:
                assert (
                    frozen_param_sum == self.frozen_param_sum
                ), f"Frozen parameters before and after training have changed! Before: {self.frozen_param_sum}, after: {frozen_param_sum}"
            except AssertionError as e:
                log.error(f"Assertion error: {str(e)}")
                raise

    @property
    def network_to_freeze(self) -> nn.Module:
        return self.network


class HashFreezeTrainerGiga(SimpleTrainerGiga, HashFreezeTrainer):
    pass


class BlockHashFreezeTrainer(HashFreezeTrainer):
    """Trainer for Block HashNet that freezes one part (MLP or hash encoding) and
    re-intialize the other to train from scratch.

    This is just like the regular `HashFreezeTrainer` if we `freeze_at == 'init'`. But
    if we `freeze_at == end_of_region0`, then it is re-defined in the context of Block
    HashNet as "freezing the MLP of a block after it trains on its first-ever seen
    region". In other words, if we switch from region `i` to `i+1`, then:
        1. We check that region `i` touches / invokes a set of blocks [j, k, l]
        2. For block j, k, l, we freeze their respective MLPs for the rest of the training.
        3. We now start fitting region `i+1`.

    To do this, we keep track of whether each block is training for the first time. We
    then set the `network_to_freeze` property accordingly.
    """

    def __init__(self, cfg: DictConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.block_to_freeze = None
        # When a block *finishes* training a region (about to switch to the next region),
        # we set it to True.
        self.block_has_trained = torch.zeros(
            len(self.network.block_hash_nets), dtype=torch.bool, device=self.device
        )

    def freeze_part_of_network(
        self,
        reinit_unfrozen_part: bool,
        blocks_to_freeze: Optional[List[int]] = None,
    ) -> None:
        """See super class for documentation. Unlike the default for super class, we do
        not re-initialize the optimizer when freezing, which makes it convenient to call
        freeze sequentially for a list of blocks.

        Args:
            blocks_to_freeze: List of blocks to freeze. If None, freeze the entire network.
        """
        if blocks_to_freeze is None:
            self.block_to_freeze = None
            return super().freeze_part_of_network(
                reinit_unfrozen_part=reinit_unfrozen_part, reinit_optimizer=False
            )

        for block_i in blocks_to_freeze:
            log.info(f"Select block {block_i} of network to freeze...")
            self.block_to_freeze = block_i
            super().freeze_part_of_network(
                reinit_unfrozen_part=reinit_unfrozen_part, reinit_optimizer=False
            )

    def maybe_switch_region(
        self, model_input: Tensor, ground_truth: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Overwrite to freeze part of the network when appropriate."""
        if self.freeze_at == "end_of_region0" and (
            self.need_to_switch_region and not torch.all(self.block_has_trained)
        ):
            # Get the list of blocks touched by the current region
            # NOTE: This may be problematic if the region size is huge and moving all
            # input coordinates to device is wasteful.
            blocks_touched = torch.unique(
                self.network.coords_to_block_indices(
                    **move_to(self.dataset.input, self.device)
                )
            )

            blocks_to_freeze = []
            for block_i in blocks_touched:
                if not self.block_has_trained[block_i]:
                    # We have touched a block that has not trained a complete region
                    # Freeze it and mark it as trained
                    blocks_to_freeze.append(block_i)
                    self.block_has_trained[block_i] = True

            self.freeze_part_of_network(
                reinit_unfrozen_part=False, blocks_to_freeze=blocks_to_freeze
            )

        return super().maybe_switch_region(model_input, ground_truth)

    @property
    def network_to_freeze(self) -> nn.Module:
        if self.block_to_freeze is None:
            return self.network
        else:
            return self.network.block_hash_nets[self.block_to_freeze]
