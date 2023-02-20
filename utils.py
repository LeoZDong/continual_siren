from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
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

    # This would result in duplicate module names, e.g. net.0.linear.weight and
    # net.0.linear.bias would both append module name `net.0.linear`!
    # Remove duplicates
    module_names = sorted(set(module_names))

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


def get_vertex_indices(min_vertex_indices: Tensor) -> Tensor:
    """Get vertex indices of a voxel / pixel given its "minimum" (bottom left) index.
    For example, in 2D, if `min_vertex_indices = (0, 0)`, then the pixel indices are:
        (0, 0), (0, 1), (1, 0), (1, 1),
    in this order.
    Args:
        min_vertex_indices: (bsz, 2 or 3) Batch of minimum vertex indices.

    Returns:
        voxel_indices: (bsz, 4, 2) or (bsz, 8, 3) Batch of voxel / pixel indices.
    """
    device = min_vertex_indices.device
    if min_vertex_indices.shape[1] == 2:
        shift = torch.tensor([[[i, j] for i in [0, 1] for j in [0, 1]]], device=device)
    else:
        shift = torch.tensor(
            [[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]], device=device
        )

    return min_vertex_indices.unsqueeze(1) + shift


def get_adjacent_vertices(
    x: Tensor,
    grid_min: float,
    grid_max: float,
    vertex_resolution: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Get adjacent vertices for a batch of coordinates in a grid of some resolution.
    In 2D, each coordinate has 4 adjacent vertices; in 3D, each coordinate has 8.
    Args:
        x: (bsz, 2 or 3) Batch of coordinates.
        grid_min: Possible min value of coordinates x. This is usually -1 or 0.
        grid_max: Possible max value of coordinates x. This is usually 1.
        vertex_resolution: Number of vertices on each side of this grid. This is
            different from the definition in the paper (where `resolution` is the number
            of voxels on each side), but consistent with the official CUDA implementation.
            The number of voxels on each side is simply `resolution - 1`.

    Returns:
        vertex_indices: (bsz, 4, 2) or (bsz, 8, 3) Indices of adjacent vertices.
        min_vertex_coords: (bsz, 2 or 3) "Minimum" (bottom left) of adjacent vertices.
        max_vertex_coords: (bsz, 2 or 3) "Maximum" (top right) of adjacent vertices.
        NOTE: min_vertex_coord and max_vertex_coord are useful for subsequent interpolation.
    """
    voxel_resolution = vertex_resolution - 1

    grid_size = grid_max - grid_min
    # For 2D, this refers to pixel size
    voxel_size = grid_size / voxel_resolution

    # NOTE: In InstantNGP paper, the equation is floor(x * resolution) because x is
    # assumed to be in [0, 1]. If x is in [-1, 1], need to shift and scale it first.
    min_vertex_indices = torch.floor(
        ((x - grid_min) / grid_size) * voxel_resolution
    ).int()
    # Need to deal with edge case because coordinate value may be 1 in our code base,
    # where in InstantNGP it is strictly less than 1. If coord = 1, `floor` will not
    # round down one index, so need to manually clip min vertex index.
    min_vertex_indices = torch.min(
        min_vertex_indices,
        torch.ones_like(min_vertex_indices) * (int(voxel_resolution) - 1),
    )
    max_vertex_indices = min_vertex_indices + 1

    # Get vertex indices of the surrounding pixel / voxel
    vertex_indices = get_vertex_indices(min_vertex_indices)

    # Recover coordinates of min / max vertices from indices
    min_vertex_coords = min_vertex_indices * voxel_size + grid_min
    max_vertex_coords = max_vertex_indices * voxel_size + grid_min

    return vertex_indices, min_vertex_coords, max_vertex_coords


def linear_interpolate(
    x: Tensor,
    min_vertex_coords: Tensor,
    max_vertex_coords: Tensor,
    vertex_values: Tensor,
) -> Tensor:
    """Linear interpolation for either 2D (bilinear) or 3D (trilinear).
    Args:
        x: (bsz, 2 or 3) Batch of coordinates to interpolate.
        min_vertex_coords: (bsz, 2 or 3) "Minimum" (bottom left) coordinates of vertices
            adjacent to `x`.
        max_vertex_coords: (bsz, 2 or 3) "Maximum" (top right) coordinates of vertices
            adjacent to `x`.
        vertex_values: (bsz, 4, dim) or (bsz, 8, dim) Values of adjacent vertices to interpolate.

    Returns:
        (bsz, dim): Interpolated values for coordinates `x`.
    """
    if x.shape[1] == 2:
        return linear_interpolate_2D(
            x, min_vertex_coords, max_vertex_coords, vertex_values
        )
    else:
        return linear_interpolate_3D(
            x, min_vertex_coords, max_vertex_coords, vertex_values
        )


def linear_interpolate_2D(
    x: Tensor,
    min_vertex_coords: Tensor,
    max_vertex_coords: Tensor,
    vertex_values: Tensor,
) -> Tensor:
    """Bilineaer interpolation.
    From the figure given in https://en.wikipedia.org/wiki/Bilinear_interpolation,
    vertices in `vertex_values` are ordered as:
        [Q11, Q12, Q21, Q22]
    """
    # TODO: We can optionally offset x by 0.5 * (max_vertex_coords - min_vertex_coord),
    # i.e. half of the voxel size, to prevent zero gradients when we perfectly align
    # with the grid vertices.
    weights_left = (x - min_vertex_coords) / (max_vertex_coords - min_vertex_coords)
    weights_right = (max_vertex_coords - x) / (max_vertex_coords - min_vertex_coords)

    # Interpolate along the first axis
    c0 = (
        weights_right[:, 0, None] * vertex_values[:, 0, :]
        + weights_left[:, 0, None] * vertex_values[:, 2, :]
    )
    c1 = (
        weights_right[:, 0, None] * vertex_values[:, 1, :]
        + weights_left[:, 0, None] * vertex_values[:, 3, :]
    )

    # Interpolate along the second axis
    c = weights_right[:, 1, None] * c0 + weights_left[:, 1, None] * c1

    return c


def linear_interpolate_3D(
    x: Tensor,
    min_vertex_coords: Tensor,
    max_vertex_coords: Tensor,
    vertex_values: Tensor,
) -> Tensor:
    """Trilinear interpolation.
    From the figure given in https://en.wikipedia.org/wiki/Trilinear_interpolation,
    vertices in `vertex_values` are ordered as:
        [c000, c001, c010, c011, c100, c101, c110, c111]
    """
    raise NotImplementedError
