import os

import hydra
import skimage.io as io
import torch
from torch import Tensor
from torch.utils.data import Dataset

bsds300_dir = f"{hydra.utils.get_original_cwd()}/BSDS300/images"


def get_mgrid(side_length: int, dim: int = 2) -> Tensor:
    """Generate a flattened grid of (x, y, ...) coordinates in a range of -1 to 1.
    Args:
        side_length: Side length of the grid (i.e. number of points).
        dim: Dimension of the grid.
    Returns:
        (side_length, side_length, dim) A tensor of `dim`-dimensional coordinates.
    """
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=side_length)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    return mgrid


def get_bsds_tensor_square(side_length: int, split: str, id: str) -> Tensor:
    """Return a BSDS image as a tensor, cropped into a square.
    Args:
        side_length: Side length of the square to crop the image into.
        split: Split of BSDS image.
        id: ID of the BSDS image.
    """
    dir = os.path.join(bsds300_dir, split, f"{id}.jpg")
    img = torch.tensor(io.imread(dir) / 255).float()

    # Resize image
    crop_x_l = (img.shape[0] - side_length) // 2
    crop_x_r = side_length + crop_x_l
    crop_y_l = (img.shape[1] - side_length) // 2
    crop_y_r = side_length + crop_y_l
    img = img[crop_x_l:crop_x_r, crop_y_l:crop_y_r, :]

    return img


class ImageFitting(Dataset):
    """Dataset that represents a single image for SIREN fitting. One (x, y) data point
    represents one coordinate and its corresponding pixel value in the image.
    """

    def __init__(self, side_length: int, data_split: str, id: str, divide_side_n: int):
        """Initialize an image fitting dataset.
        Args:
            side_length: Side length of image (i.e. number of pixels on each side).
            data_split: Data split of the BSDS dataset.
            id: ID of the single image to use.
            divide_side_n: Evenly divide each side into n segments. This results in n^2
                regions to fit sequentially if using continual learning.
        """
        super().__init__()
        spatial_dim = 2

        # Load full image and its coordinates
        self.full_pixels = get_bsds_tensor_square(side_length, data_split, id)
        self.full_coords = get_mgrid(side_length, spatial_dim)

        # Split the image and coordinates into regions
        side_length_per_region = side_length // divide_side_n
        if side_length_per_region * divide_side_n != side_length:
            print(
                f"Warning: divide_side_n={divide_side_n} does not divide side length {side_length}."
                "Some parts of the image will not belong ot any divided region!"
            )

        self.pixels_regions = []
        self.coords_regions = []
        for i in range(divide_side_n):
            for j in range(divide_side_n):
                pixels_region = self.full_pixels[
                    i * side_length_per_region : (i + 1) * side_length_per_region,
                    j * side_length_per_region : (j + 1) * side_length_per_region,
                    :,
                ]
                coords_region = self.full_coords[
                    i * side_length_per_region : (i + 1) * side_length_per_region,
                    j * side_length_per_region : (j + 1) * side_length_per_region,
                    :,
                ]
                self.pixels_regions.append(pixels_region.reshape(-1, 3))
                self.coords_regions.append(coords_region.reshape(-1, spatial_dim))

        # Flatten spatial dimension to obtain a batch of coordinates for training
        self.full_pixels = self.full_pixels.reshape(-1, 3)
        self.full_coords = self.full_coords.reshape(-1, spatial_dim)

        # Initialize current region to be the first region
        self.set_cur_region(0)

    def set_cur_region(self, region: int) -> None:
        self.cur_region = region
        self.pixels = self.pixels_regions[self.cur_region]
        self.coords = self.coords_regions[self.cur_region]

    @property
    def num_regions(self) -> int:
        return len(self.pixels_regions)

    def __len__(self):
        # return len(self.pixels_regions)
        return self.pixels.shape[0]

    def __getitem__(self, idx):
        # if idx > 0:
        # raise IndexError

        return self.coords, self.pixels
        # return self.coords[idx], self.pixels[idx]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Display image
    img = get_bsds_tensor_square(side_length=256, split="test", id="108005")
    print(img.shape)
    fig = plt.figure()
    plt.imshow(img)
    plt.show()
