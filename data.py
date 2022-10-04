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
    """
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=side_length)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
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
    """Dataset that represents a single image for SIREN fitting."""

    def __init__(self, side_length, split, id):
        super().__init__()
        self.pixels = get_bsds_tensor_square(side_length, split, id)
        self.coords = get_mgrid(side_length, 2)

        # Flatten spatial dimension for training
        self.pixels = self.pixels.reshape(-1, 3)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        return self.coords, self.pixels


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Display image
    img = get_bsds_tensor_square(side_length=256, split="test", id="108005")
    print(img.shape)
    fig = plt.figure()
    plt.imshow(img)
    plt.show()
