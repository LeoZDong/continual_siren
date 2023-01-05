from typing import Dict

import torch
from omegaconf import DictConfig
from torch import Tensor

from trainers.simple_trainer import SimpleTrainer


class RegularizeTrainer(SimpleTrainer):
    """Trainer that regularizes changes to 'important' parameters to alleviate forgetting.
    This class uses the zero importance score (has no effect). Used as base class
    for regularization-based methods that instantiate different importance scores.
    """

    def __init__(self, cfg: DictConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.reg_coef = cfg.trainer.reg_coef

        # Reference values of parameters (i.e. regularize deviations from the reference)
        # There may be multiple reference models (e.g. one per region)
        self.reference_model = []

        # Importance scores of the parameters.
        # There may be multiple importance scores.
        self.importance = []

    def set_current_model_as_ref(self) -> None:
        """Set the current model as next reference model."""
        cur_model = {
            name: param.clone().detach()
            for name, param in self.siren.named_parameters()
            if param.requires_grad
        }

        if isinstance(self.reference_model, list):
            # There are multiple reference models. Append current model as next one.
            self.reference_model.append(cur_model)
        else:
            # Set the current model as *the* reference model.
            self.reference_model = cur_model

    def calculate_and_set_importance(self, **kwargs) -> Dict[str, Tensor]:
        """Calculate the importance of each parameter. As a base class, we use the zero
        importance so no regularization is done. Inherited classes implement this method
        differently.
        """
        raise NotImplementedError

    def get_reg_loss(
        self, ref_params: Dict[str, Tensor], importance: Dict[str, Tensor]
    ) -> Tensor:
        """Calculate the regularization loss for **one** pair of reference params and
        importance scores. To be instantated in subclasses.

        Args:
            ref_params: Reference parameters keyed by param name. We penalize deviations
                of the current model from reference parameters.
            importance: Importance of each parameter keyed by param name. Importance of
                a tensor parameter is a tensor with the same shape.
        """
        raise NotImplementedError

    def loss(
        self,
        model_output: Tensor,
        ground_truth: Tensor,
        regularize: bool = True,
        **kwargs,
    ) -> Tensor:
        """MSE loss with regularization. This procedure is generally universal among all
        regularization-based methods.
        """
        loss = super().loss(model_output, ground_truth)

        if regularize and len(self.reference_model) > 0:
            reg_loss = 0

            if isinstance(self.reference_model, list):
                # There are multiple reference models. Regularize for each one.
                for i, ref_params in enumerate(self.reference_model):
                    importance = (
                        self.importance[i]
                        if isinstance(self.importance, list)
                        else self.importance
                    )
                    reg_loss += self.get_reg_loss(ref_params, importance)
            else:
                # If there is only one reference model, there must be only one importance
                # score, so we do not need to deal with that case.
                reg_loss += self.get_reg_loss(self.reference_model, self.importance)

            loss += self.reg_coef * reg_loss

        return loss
