import numpy as np


def rgb_float2uint(rgb: np.ndarray):
    """Convert a float array representing RGB values to unsigned int aray.
    Args:
        rgb: Float array with values expected to be in range [0, 1].
    """
    return (np.clip(rgb, a_min=0, a_max=1) * 255).astype(np.uint8)