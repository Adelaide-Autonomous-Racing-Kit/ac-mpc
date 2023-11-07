from typing import Dict

from .localisation import LocaliseOnTrack


class LocalisationTracker:
    def __init__(self, localiser: LocaliseOnTrack, gt_poses: Dict):
        self._localiser = localiser
        self._n_total_steps = 0
        self._n_steps = 0
        self._n_resets = 0
        self._previous_localiser_state = False
        self._n_steps_localised_for = []
        self._n_steps_to_convergence = []
        self._execution_times = []
        self._errors = {"x": [], "y": [], "yaw": []}
        self._gt_poses = gt_poses

    def update(self, time_to_step: float):
        self._record_execution_time(time_to_step)
        self._check_for_reset()
        self._check_for_convergence()
        self._calculate_error()
        self._step()

    def _record_execution_time(self, time: float):
        self._execution_times.append(time)

    def _check_for_reset(self):
        if self._has_reset():
            self._n_steps_localised_for.append(self._n_steps)
            self._n_resets += 1
            self._n_steps = 0

    def _has_reset(self) -> bool:
        return self._previous_localiser_state and not self._localiser.localised

    def _check_for_convergence(self):
        if self._has_converged:
            self._n_steps_to_convergence.append(self._n_steps)

    def _has_converged(self) -> bool:
        return self._localiser.localised and not self._previous_localiser_state

    def _calculate_error(self):
        if self._localiser.localised:
            estimated_pose, _, _, _ = self._localiser.estimated_position
            pose = self._gt_poses[self._n_total_steps]
            self._errors["x"].append(pose["x"] - estimated_pose[0])
            self._errors["y"].append(pose["y"] - estimated_pose[1])
            self._errors["yaw"].append(pose["yaw"] - estimated_pose[2])

    def _step(self):
        self._previous_localiser_state = self._localiser.localised
        self._n_steps += 1
        self._n_total_steps += 1
