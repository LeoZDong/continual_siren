import numpy as np
import pytest
import torch

import utils


@pytest.fixture(scope="class")
def prepare_hard_coded_values_adj(request):
    case_simple_2D = {
        "x": torch.tensor([[0.5, 0.5]]),
        "vertex_indices_expected": torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]]),
        "min_vertex_coords_expected": torch.tensor([[0, 0]]),
        "max_vertex_coords_expected": torch.tensor([[1, 1]]),
        "grid_min": 0,
        "grid_max": 1,
        "resolution": 1,
    }

    case_quadrant_2D_1 = {
        "x": torch.tensor([[0.25, 0.25]]),
        "vertex_indices_expected": torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]]),
        "min_vertex_coords_expected": torch.tensor([[0, 0]]),
        "max_vertex_coords_expected": torch.tensor([[0.5, 0.5]]),
        "grid_min": 0,
        "grid_max": 1,
        "resolution": 2,
    }

    case_quadrant_2D_2 = {
        "x": torch.tensor([[0.75, 0.25]]),
        "vertex_indices_expected": torch.tensor([[[1, 0], [1, 1], [2, 0], [2, 1]]]),
        "min_vertex_coords_expected": torch.tensor([[0.5, 0]]),
        "max_vertex_coords_expected": torch.tensor([[1, 0.5]]),
        "grid_min": 0,
        "grid_max": 1,
        "resolution": 2,
    }

    request.cls.hard_coded_values = [
        case_simple_2D,
        case_quadrant_2D_1,
        case_quadrant_2D_2,
    ]


@pytest.fixture(scope="class")
def prepare_voxel_ref_implementation(request):
    def get_voxel_vertices(xyz, resolution):
        """Reference voxel implementation from the HashNeRF repo."""
        BOX_OFFSETS = torch.tensor(
            [[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]]
        )

        box_min, box_max = torch.tensor([-1, -1, -1]), torch.tensor([1, 1, 1])

        grid_size = (box_max - box_min) / resolution

        bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
        voxel_min_vertex = bottom_left_idx * grid_size + box_min
        voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0, 1.0, 1.0]) * grid_size

        voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS

        return voxel_min_vertex, voxel_max_vertex, voxel_indices

    x = torch.rand((10, 3))
    resolution = 16
    min_vertex_coords, max_vertex_coords, vertex_indices = get_voxel_vertices(
        x, resolution
    )
    case_ref_impl = {
        "x": x,
        "vertex_indices_expected": vertex_indices,
        "min_vertex_coords_expected": min_vertex_coords,
        "max_vertex_coords_expected": max_vertex_coords,
        "grid_min": -1,
        "grid_max": 1,
        "resolution": resolution,
    }
    request.cls.hard_coded_values.append(case_ref_impl)


@pytest.mark.usefixtures(
    "prepare_hard_coded_values_adj", "prepare_voxel_ref_implementation"
)
class TestAdjVertices:
    def test_hard_coded_values(self):
        for test_case in self.hard_coded_values:

            (
                vertex_indices,
                min_vertex_coords,
                max_vertex_coords,
            ) = utils.get_adjacent_vertices(
                test_case["x"],
                grid_min=test_case["grid_min"],
                grid_max=test_case["grid_max"],
                resolution=test_case["resolution"],
            )

            assert torch.all(
                vertex_indices == test_case["vertex_indices_expected"]
            ), f"Got vertex_indices={vertex_indices}, but expected to be {test_case['vertex_indices_expected']}"
            assert torch.all(
                min_vertex_coords == test_case["min_vertex_coords_expected"]
            ), f"Got min_vertex_coords={min_vertex_coords}, but expected to be {test_case['min_vertex_coords_expected']}"
            assert torch.all(
                max_vertex_coords == test_case["max_vertex_coords_expected"]
            ), f"Got min_vertex_coords={max_vertex_coords}, but expected to be {test_case['max_vertex_coords_expected']}"


class TestInterpl:
    def test_relative(self):
        """Sanity check test that interpolated values should be closer to nearest values."""
        vertex_values = torch.tensor([0, 1, 2, 3]).unsqueeze(0).unsqueeze(-1)
        # Coordinate close to bottom left (with vertex value 0)
        x_bl = torch.tensor([[0.1, 0.1]])
        # Coordinate close to top right (with vertex value 3)
        x_tr = torch.tensor([[0.9, 0.9]])

        min_vertex_coords = torch.tensor([[0, 0]])
        max_vertex_coords = torch.tensor([[1, 1]])

        bl = utils.linear_interpolate(
            x_bl, min_vertex_coords, max_vertex_coords, vertex_values
        )
        tr = utils.linear_interpolate(
            x_tr, min_vertex_coords, max_vertex_coords, vertex_values
        )

        print(f"Interpolated value at bottom left: {bl.item()}")
        print(f"Interpolated value at top right: {tr.item()}")

        # NOTE: The above example is also consistent with this online interpolation
        # calculator: https://www.omnicalculator.com/math/bilinear-interpolation

        assert bl < tr
