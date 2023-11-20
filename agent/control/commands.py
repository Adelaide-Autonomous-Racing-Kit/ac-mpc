from typing import List

import numpy as np
from loguru import logger


class TemporalCommandInterpolator:
    def __init__(self):
        self._cum_time = np.array([])
        self._commands = np.array([])

    def _get_closet_command_index(self, elapsed_time: float) -> int:
        distances = self._calculate_temporal_distances(elapsed_time)
        index = np.argmin(abs(distances))
        logger.info(f"{distances}")
        logger.info(f"{index}")
        return index, distances[index]

    def _calculate_temporal_distances(self, elapsed_time: float) -> np.array:
        return self._cum_time - elapsed_time

    def get_command(self, elapsed_time: float) -> np.array:
        index_a, index_b = self._get_indices_of_commands_to_interpolate(elapsed_time)
        return self._interpolate_command(index_a, index_b, elapsed_time)

    def _get_indices_of_commands_to_interpolate(self, elapsed_time: float) -> List[int]:
        index_a, distance = self._get_closet_command_index(elapsed_time)
        if index_a == 0 or index_a == (len(self._commands) - 1):
            index_b = index_a
            return index_a, index_b
        if distance < 0:
            index_b = index_a + 1
        else:
            index_b = index_a - 1
        return index_a, index_b

    def _interpolate_command(
        self,
        index_a: int,
        index_b: int,
        elapsed_time: float,
    ) -> np.array:
        if index_a == index_b:
            return self._commands[index_a]
        x_a, y_a = self._cum_time[index_a], self._commands[index_a]
        x_b, y_b = self._cum_time[index_b], self._commands[index_b]
        # Linear Interpolation
        portion_a = (x_b - elapsed_time) / (x_b - x_a)
        portion_b = (elapsed_time - x_a) / (x_b - x_a)
        return y_a * portion_a + y_b * portion_b
