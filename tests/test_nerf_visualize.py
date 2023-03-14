"""Test NeRF dataset by visualizing camera poses."""

import plotly.graph_objs as go
import torch

from data.nerf_data import NeRFSyntheticDataset
from data.region_simulators import NeRFRegionSimulator


def main():
    region_simulator = NeRFRegionSimulator(
        num_images_per_region=1, img_h=800, img_w=800
    )

    dataset = NeRFSyntheticDataset(
        path="datasets/nerf_synthetic/lego",
        split="train",
        downsample=1,
        region_simulator=region_simulator,
    )

    # Create a 3D scatter plot of arros
    fig = go.Figure()

    # Plot training camera poses
    inputs = dataset.input_regions
    for i in range(len(inputs)):
        # Pick the first ray of the image to represent the pose of this entire image
        ray_o = inputs[i]["rays_o"][0]
        ray_d = inputs[i]["rays_d"][0]

        # Add camera poses as arrows
        fig.add_trace(
            go.Cone(
                x=[ray_o[0]],
                y=[ray_o[1]],
                z=[ray_o[2]],
                u=[ray_d[0]],
                v=[ray_d[1]],
                w=[ray_d[2]],
                sizemode="scaled",
                sizeref=0.1,
                showscale=False,
                anchor="tip",
                colorscale="Blues",
            )
        )

    # Plot test camera poses
    inputs = dataset.video_inputs
    for i in range(len(inputs)):
        # Pick the first ray of the image to represent the pose of this entire image
        ray_o = inputs[i]["rays_o"][0]
        ray_d = inputs[i]["rays_d"][0]

        fig.add_trace(
            go.Cone(
                x=[ray_o[0]],
                y=[ray_o[1]],
                z=[ray_o[2]],
                u=[ray_d[0]],
                v=[ray_d[1]],
                w=[ray_d[2]],
                sizemode="scaled",
                sizeref=0.1,
                showscale=False,
                anchor="tip",
                colorscale="Reds",
            )
        )

    # Set the layout
    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

    # Show the plot
    fig.show()


if __name__ == "__main__":
    main()
