from typing import Dict, Tuple

from omegaconf import DictConfig
from torch import Tensor

from trainers.regularize_trainer import RegularizeTrainer


class EWCTrainer(RegularizeTrainer):
    """Trainer that regularizes changes to 'important' parameters, where importance is
    measured by EWC.
    """

    def __init__(self, cfg: DictConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        # In EWC, we have multiple reference models and multiple importance scores, one
        # for each task (i.e. region).
        self.reference_model = []
        self.importance = []

    def get_reg_loss(
        self, ref_params: Dict[str, Tensor], importance: Dict[str, Tensor]
    ) -> Tensor:
        """See base class for description."""
        model_params = dict(self.siren.named_parameters())

        reg_loss = 0
        for name in ref_params.keys():
            reg_loss += (
                importance[name] * ((model_params[name] - ref_params[name]) ** 2)
            ).sum()

        return reg_loss

    def maybe_switch_region(
        self, model_input: Tensor, ground_truth: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if self.need_to_switch_region:
            # Set importance and reference model when we are about to switch to the
            # next region. At this point, `model_input` and `ground_truth` still
            # correspond to the **old** region.
            self.calculate_and_set_importance(
                model_input=model_input, ground_truth=ground_truth
            )
            self.set_current_model_as_ref()

        return super().maybe_switch_region(model_input, ground_truth)

    def calculate_and_set_importance(
        self, model_input: Tensor, ground_truth: Tensor
    ) -> None:
        """In EWC, importance is the Fisher information diagonal, which is approximated
        as the square gradient with respect to the previous task / region. We will never
        call `calculate_and_set_importance` when we are at the first region.

        Args:
            model_input: Input coordinates of the previous region.
            ground_truth: Pixel values of the previous region.
        """
        model_output = self.siren(model_input)[0]

        loss = self.loss(model_output, ground_truth, regularize=False)
        self.siren.zero_grad()
        loss.backward()

        importance = {}
        for name, param in self.siren.named_parameters():
            if param.requires_grad:
                importance[name] = param.grad**2

        # Because EWC has multiple importance scores, append as the next importance score.
        self.importance.append(importance)
