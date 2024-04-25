from threading import local
from typing import Dict

import numpy as np
from loguru import logger

from localisation.benchmarking.test_localiser import TestLocaliser


class LocalisationTracker:
    def __init__(self, localiser: TestLocaliser, gt_poses: Dict):
        self._localiser = localiser
        self._n_total_observations = 0
        self._n_steps = 0
        self._n_total_steps = 0
        self._n_resets = 0
        self._previous_localiser_state = False
        self._n_steps_localised_for = []
        self._n_steps_to_convergence = []
        self._observation_execution_times = []
        self._step_execution_times = []
        self._errors = {"x": [], "y": [], "yaw": []}
        self._gt_poses = gt_poses

    def update_step(self, time_to_step: float):
        self._record_step_execution_time(time_to_step)
        self._calculate_error()
        self._step()

    def update_observation(self, time_to_step: float):
        self._record_observation_execution_time(time_to_step)
        self._check_for_reset()
        self._check_for_convergence()
        self._previous_localiser_state = self._localiser.is_localised
        self._n_total_observations += 1

    def _record_step_execution_time(self, time: float):
        self._step_execution_times.append(time)

    def _record_observation_execution_time(self, time: float):
        self._observation_execution_times.append(time)

    def _check_for_reset(self):
        if self._has_reset():
            self._n_steps_localised_for.append(self._n_steps)
            self._n_resets += 1
            self._n_steps = 0

    def _has_reset(self) -> bool:
        return self._previous_localiser_state and not self._localiser.is_localised

    def _check_for_convergence(self):
        if self._has_converged():
            self._n_steps_to_convergence.append(self._n_steps)
            self._n_steps = 0

    def _has_converged(self) -> bool:
        return self._localiser.is_localised and not self._previous_localiser_state

    def _calculate_error(self):
        if self._localiser.is_localised:
            estimated_pose = self._localiser.estimated_position
            pose = self.current_ground_truth_pose
            self._errors["x"].append(pose["x"] - estimated_pose[0])
            self._errors["y"].append(pose["y"] - estimated_pose[1])
            rotation_error = pose["yaw"] - estimated_pose[2]
            rotation_error = (rotation_error + np.pi) % (2 * np.pi) - np.pi
            self._errors["yaw"].append(rotation_error)

    @property
    def current_ground_truth_pose(self) -> Dict:
        pose = self._gt_poses[self._n_total_steps][0]
        return {"x": -1.0 * pose[0], "y": pose[2], "yaw": pose[3]}

    def _step(self):
        self._n_steps += 1
        self._n_total_steps += 1

    def average_position_error(self) -> np.array:
        return np.mean(np.abs(self._errors["x"]) + np.abs(self._errors["y"]))

    def average_rotation_error(self) -> np.array:
        return np.mean(np.abs(self._errors["yaw"]))

    def percentage_of_steps_localised_for(self) -> float:
        localised_steps = sum(self._n_steps_localised_for)
        localised_steps += self._n_steps
        return 100 * (localised_steps / self._n_total_steps)
