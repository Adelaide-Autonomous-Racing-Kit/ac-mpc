import ctypes
import multiprocessing as mp
from typing import Dict

import numpy as np


class SharedPose:
    def __init__(self):
        self.__setup()

    def __setup(self):
        # [x, y, z, roll, pitch, yaw]
        self._shared_pose = SharedArray(ctypes.c_float, 6)

    @property
    def pose_dict(self) -> Dict:
        pose = self._shared_pose.array
        pose_dict = {
            "x": pose[0],
            "y": pose[1],
            "z": pose[2],
            "roll": pose[3],
            "pitch": pose[4],
            "yaw": pose[5],
        }
        return pose_dict

    @property
    def pose(self) -> np.array:
        pose = self._shared_pose.array
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
        self._shared_pose.array = pose


class SharedString:
    def __init__(self, n_char: int):
        self.__setup(n_char)

    def __setup(self, n_char: int):
        self._mp_string = mp.Array(ctypes.c_wchar, n_char)

    @property
    def value(self) -> str:
        with self._mp_string.get_lock():
            value = self._mp_string.value
        return value

    @value.setter
    def value(self, value: str):
        with self._mp_string.get_lock():
            self._mp_string.value = value


class SharedArray:
    def __init__(self, dtype: ctypes, n_values: int):
        self.__setup(dtype, n_values)

    def __setup(self, dtype: ctypes, n_values: int):
        mp_array = mp.Array(dtype, n_values)
        np_array = np.ndarray(
            [n_values],
            dtype=dtype,
            buffer=mp_array.get_obj(),
        )
        self._shared_state_buffer = (mp_array, np_array)

    @property
    def array(self) -> np.array:
        mp_array, np_array = self._shared_state_buffer
        with mp_array.get_lock():
            array = np_array.copy()
        return array

    @array.setter
    def array(self, array: np.array):
        mp_array, np_array = self._shared_state_buffer
        with mp_array.get_lock():
            np_array[:] = array


class SharedSessionDetails:
    def __init__(self):
        self.__setup()

    def __setup(self):
        self._session_info = SharedArray(ctypes.c_int, 6)
        self._session_info.array = np.zeros((6))

    @property
    def current_laptime(self) -> int:
        return self._session_info.array[0]

    @property
    def last_laptime(self) -> int:
        return self._session_info.array[1]

    @property
    def best_laptime(self) -> int:
        return self._session_info.array[2]

    @property
    def last_sector_time(self) -> int:
        return self._session_info.array[3]

    @property
    def n_laps_completed(self) -> int:
        return self._session_info.array[4]

    @property
    def current_sector(self) -> int:
        return self._session_info.array[5]

    @property
    def session_info(self) -> np.array:
        return self._session_info.array

    @session_info.setter
    def session_info(self, session_info: dict):
        session_info = np.array(
            [
                session_info["i_current_time"],
                session_info["i_last_time"],
                session_info["i_best_time"],
                session_info["last_sector_time"],
                session_info["completed_laps"],
                session_info["current_sector_index"],
            ]
        )
        self._session_info.array = session_info
