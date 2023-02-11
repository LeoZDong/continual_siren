import logging

from omegaconf import DictConfig

from trainers.ewc_trainer import EWCTrainer

log = logging.getLogger(__name__)


class HashEWCTrainer(EWCTrainer):
    """Trainer that uses EWC regularization on either the MLP part, the hash encoding
    part, all the entire HashNet.
    """

    def __init__(self, cfg: DictConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        # Exclude parameters for regularization if their name start with strings
        # in `reg_exclude_param_names`.
        self.reg_exclude_param_names = cfg.trainer.reg_exclude_param_names

    def set_current_model_as_ref(self) -> None:
        """Set the current model as next reference model."""
        cur_model = {
            name: param.clone().detach()
            for name, param in self.network.named_parameters()
            if param.requires_grad
            and not any(
                name.startswith(exclude_name)
                for exclude_name in self.reg_exclude_param_names
            )
        }

        if isinstance(self.reference_model, list):
            # There are multiple reference models. Append current model as next one.
            self.reference_model.append(cur_model)
        else:
            # Set the current model as *the* reference model.
            self.reference_model = cur_model
