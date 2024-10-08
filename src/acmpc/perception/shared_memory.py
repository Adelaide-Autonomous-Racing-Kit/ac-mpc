import ctypes
import multiprocessing as mp
from typing import Tuple

import numpy as np


class SharedImage:
    def __init__(self, height: int, width: int, channels: int):
        self._image_shape = (height, width, channels)
        self.__setup()

    def __setup(self):
        mp_array = mp.Array(ctypes.c_uint8, self._n_pixels)
        np_array = np.ndarray(
            self._image_shape,
            dtype=np.uint8,
            buffer=mp_array.get_obj(),
        )
        self._shared_image_buffer = (mp_array, np_array)
        self._is_stale = mp.Value("i", True)

    @property
    def _n_pixels(self):
        return int(np.prod(self._image_shape))

    @property
    def image(self) -> np.array:
        image_mp_array, image_np_array = self._shared_image_buffer
        with image_mp_array.get_lock():
            image = image_np_array.copy()
        self.is_stale = True
        return image

    @image.setter
    def image(self, image: np.array):
        image_mp_array, image_np_array = self._shared_image_buffer
        with image_mp_array.get_lock():
            image_np_array[:] = image
        self.is_stale = False

    @property
    def fresh_image(self) -> np.array:
        self._wait_for_fresh_image()
        return self.image

    def _wait_for_fresh_image(self):
        while self.is_stale:
            continue

    @property
    def is_stale(self) -> bool:
        """
        Checks if the current image has been read by any consumer

        :return: True if the image has been read, false if it has not
        :rtype: bool
        """
        with self._is_stale.get_lock():
            is_stale = self._is_stale.value
        return is_stale

    @is_stale.setter
    def is_stale(self, is_stale: bool):
        """
        Sets the flag indicating if the image has been read previously

        :is_stale: True if the image has been read, false if it has not
        :type is_stale: bool
        """
        with self._is_stale.get_lock():
            self._is_stale.value = is_stale


class SharedPoints:
    def __init__(self, n_points: np.array, n_dimensions: int):
        self._n_points = n_points
        self._n_dimensions = n_dimensions
        self.__setup()

    @property
    def points(self) -> np.array:
        point_mp_array, point_np_array = self._shared_image_buffer
        with point_mp_array.get_lock():
            points = point_np_array.copy()
        return points

    @points.setter
    def points(self, points: np.array):
        point_mp_array, point_np_array = self._shared_image_buffer
        with point_mp_array.get_lock():
            point_np_array[:] = points

    def __setup(self):
        self._setup_shared_array()

    def _setup_shared_array(self):
        mp_array = mp.Array(ctypes.c_float, self._n_values)
        np_array = np.ndarray(
            self._shape,
            dtype=np.float32,
            buffer=mp_array.get_obj(),
        )
        self._shared_image_buffer = (mp_array, np_array)

    @property
    def _n_values(self) -> int:
        n_dimensions = 1 if self._n_dimensions == 0 else self._n_dimensions
        return self._n_points * n_dimensions

    @property
    def _shape(self) -> Tuple[int, int]:
        if self._n_dimensions == 0:
            return self._n_points
        return (self._n_points, self._n_dimensions)
