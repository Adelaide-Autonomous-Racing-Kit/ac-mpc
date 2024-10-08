import cv2
import numpy as np


def flip_and_rotate(image: np.array) -> np.array:
    return cv2.rotate(flip(image), cv2.ROTATE_90_CLOCKWISE)


def flip(image: np.array) -> np.array:
    return cv2.flip(image, 0)
