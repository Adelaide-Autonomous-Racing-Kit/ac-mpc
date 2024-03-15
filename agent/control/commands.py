from __future__ import annotations
from typing import List

import numpy as np
from loguru import logger

class TemporalCommandInterpolator:
    def __init__(self, mpc: SpatialMPC):
        self._MPC = mpc
    
    @property
    def _cum_time(self) -> np.array:
        return self._MPC.cum_time
    
    @property
    def _commands(self) -> np.array:
        return self._MPC.projected_control.T
    
    def __call__(self, elapsed_time: float) -> np.array:
        return self.get_command(elapsed_time)
        
    def get_command(self, elapsed_time: float) -> np.array:
        index_a, index_b = self._get_indices_of_commands_to_interpolate(elapsed_time)
        return self._interpolate_command(index_a, index_b, elapsed_time)

    def _get_indices_of_commands_to_interpolate(self, elapsed_time: float) -> List[int]:
        index_a, distance = self._get_closet_command_index(elapsed_time)
        if self._is_start_or_end_index(index_a):
            index_b = index_a
        elif self._is_closest_command_in_the_past(distance):
            index_b = index_a + 1
        else:
            index_b = index_a - 1
        return index_a, index_b

    def _get_closet_command_index(self, elapsed_time: float) -> int:
        distances = self._calculate_temporal_distances(elapsed_time)
        index = np.argmin(abs(distances))
        return index, distances[index]

    def _calculate_temporal_distances(self, elapsed_time: float) -> np.array:
        return self._cum_time - elapsed_time

    def _is_start_or_end_index(self, index: int) -> bool:
        return index == 0 or index == (len(self._commands) - 1)

    def _is_closest_command_in_the_past(self, distance: int) -> bool:
        return distance < 0

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

class TemporalCommandSelector:
    def __init__(self, mpc: SpatialMPC):
        self._MPC = mpc
    
    @property
    def _cum_time(self) -> np.array:
        return self._MPC.cum_time
    
    @property
    def _commands(self) -> np.array:
        return self._MPC.projected_control.T
    
    def __call__(self, elapsed_time: float) -> np.array:
        return self.get_command(elapsed_time)
        
    def get_command(self, elapsed_time: float) -> np.array:
        index = self._get_closet_command_index(elapsed_time)
        command = self._commands[index]
        return command

    def _get_closet_command_index(self, elapsed_time: float) -> int:
        distances = self._calculate_temporal_distances(elapsed_time)
        index = np.argmin(abs(distances))
        if distances[index] > 0:
            index - 1
        return index

    def _calculate_temporal_distances(self, elapsed_time: float) -> np.array:
        return self._cum_time - elapsed_time
