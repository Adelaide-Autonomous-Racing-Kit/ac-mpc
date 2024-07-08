import ctypes
import multiprocessing as mp
from typing import Dict

import numpy as np


class SharedPose:
    def __init__(self):
        self.__setup()

    def __setup(self):
        # [x, y, z, roll, pitch, yaw]
        mp_array = mp.Array(ctypes.c_float, 6)
        np_array = np.ndarray(
            [6],
            dtype=np.float32,
            buffer=mp_array.get_obj(),
        )
        self._shared_state_buffer = (mp_array, np_array)

    @property
    def bev_pose(self) -> Dict:
        bev_pose = {}
        pose = self._pose
        bev_pose["x"] = -1 * pose[0]  # x flip
        bev_pose["y"] = pose[2]  # in game y is up
        bev_pose["yaw"] = pose[5]
        return bev_pose

    @property
    def _pose(self) -> np.array:
        state_mp_array, state_np_array = self._shared_state_buffer
        with state_mp_array.get_lock():
            pose = state_np_array.copy()
        return pose

    @property
    def pose(self) -> np.array:
        pose = self._pose
        return (pose[0], pose[1], pose[2], pose[5], pose[4], pose[3])

    @pose.setter
    def pose(self, observation: Dict):
        x = observation["full_pose"]["x"]
        y = observation["full_pose"]["y"]
        z = observation["full_pose"]["z"]
        roll = observation["full_pose"]["roll"]
        pitch = observation["full_pose"]["pitch"]
        yaw = observation["full_pose"]["yaw"]
        pose = np.array([x, y, z, roll, pitch, yaw])
        state_mp_array, state_np_array = self._shared_state_buffer
        with state_mp_array.get_lock():
            state_np_array[:] = pose
