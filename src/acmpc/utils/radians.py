import numpy as np


def convert_radians_to_plus_minus_pi(radians: float) -> float:
    """
    Converts a value in radians to its equivalent on the interval (-pi, pi)
    """
    return (((np.pi / 2) - radians + np.pi) % (2 * np.pi)) - np.pi
