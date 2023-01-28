from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune


def rgb_float2uint(rgb: np.ndarray):
    """Convert a float array representing RGB values to unsigned int aray.
    Args:
        rgb: Float array with values expected to be in range [0, 1].
    """
    return (np.clip(rgb, a_min=0, a_max=1) * 255).astype(np.uint8)


def mse2psnr(mse: float, max_intensity: float = 1):
    """Convert MSE to PSNR."""
    return 20 * np.log10(max_intensity) - 10 * np.log10(mse)


def get_module_names(model: nn.Module) -> List[str]:
    """Get all module names of a given `model`. Module name is everything in the param
    name except for the last word. For example, if the param name is 'net.0.linear.weight',
    then module name is 'net.0.linear'. This is used to specify parameters to prune.
    """
    module_names = []
    for param_name in dict(model.named_parameters()).keys():
        # Module name is everything in param name except for the last word
        # e.g. If param name is 'net.0.linear.weight', then module name is 'net.0.linear'
        name = ".".join(param_name.split(".")[:-1])
        module_names.append(name)
    return module_names


def get_prunable_params(model: nn.Module) -> List[Tuple[nn.Module, str]]:
    """Get all prunable parameters of a given `model`. By default, we prune both weight
    and bias of a module.
    Args:
        model: Source model from which to extract prunable parameters.

    Returns:
        List of two-tuples. Each tuple contains (module: nn.Module, param_name: str)
        that specifies the module and the param name of that module we wish to prune.
    """
    module_param_pairs = []
    module_names = get_module_names(model)
    named_modules_dict = dict(model.named_modules())
    for module_name in module_names:
        module = named_modules_dict[module_name]
        module_param_pairs.append((module, "weight"))
        module_param_pairs.append((module, "bias"))

    return module_param_pairs


def apply_pruning(
    prunable_params: List[Tuple[nn.Module, str]],
    prune_amount: float,
    finalize_pruning: bool,
) -> None:
    """Apply pruning on a specified list of prunable parameters. Use L1 unstructured
    pruning by default. TODO: Allow configuring custom pruning methods in the future.
    Args:
        model: List of two-tuples that represents the module and param name to
            prune. Expected to be the output of `get_prunable_params`.
        prune_amount: Fraction of parameters to prune.
        finalize_pruning: Whether to "finalize" pruning by removing the pruning mask and
            modifying the module parameters directly. If False, the resulting module
            will contain the original param values but stores an additional param mask;
            the mask is applied in both forward and backward pass. If True, the modules
            will contain zeroed-out params. In short, if `finalize_pruning=True`, the
            pruned out params can still be updated in a backward pass. Otherwise, they
            will not be updated in a backward pass and remain zero.
    """
    for module, prunable_param_name in prunable_params:
        prune.l1_unstructured(module, prunable_param_name, amount=prune_amount)
        if finalize_pruning:
            prune.remove(module, prunable_param_name)


def prune_model(model: nn.Module, prune_amount: float, finalize_pruning: bool) -> None:
    """Prune all parameters of `model`."""
    # Apple silicon GPU (mps) does not support prune yet. Move to CPU (may cost performance).
    device = next(model.parameters()).device
    if device.type == "mps":
        model.cpu()

    prunable_params = get_prunable_params(model)
    apply_pruning(prunable_params, prune_amount, finalize_pruning)

    if device.type == "mps":
        model.to(device)
