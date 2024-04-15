import os
import time
from typing import Dict, List

from agent.localisation.benchmarking.utils import load

from localisation.localisation import LocaliseOnTrack
from localisation.tracker import LocalisationTracker
from localisation.benchmarking.utils import LocalisationRecording
from localisation.visualisation import LocalisationVisualise
from localisation.benchmarking.test_localiser import TestLocaliser


class BenchmarkLocalisation:
    def __init__(self, cfg: Dict):
        self.__setup(cfg)

    def __setup(self, cfg: Dict):
        self._unpack_config(cfg)
        self._recording = LocalisationRecording(self._data_path)
        self._setup_particle_filter()
        self._setup_tracker(self._localiser)
        self._setup_visualiser(self._localiser, self._tracker)

    def _unpack_config(self, cfg: Dict):
        self._cfg = cfg
        self._data_path = cfg["data_path"]
        self._localisation_config = cfg["particle_filter"]

    def _setup_particle_filter(self) -> LocaliseOnTrack:
        self._localiser = TestLocaliser(self._cfg)

    def _setup_tracker(self, localiser: LocaliseOnTrack) -> LocalisationTracker:
        self._tracker = LocalisationTracker(localiser, self._gt_poses)

    def _setup_visualiser(
        self,
        localiser: LocaliseOnTrack,
        tracker: LocalisationTracker,
    ):
        self._visualiser = LocalisationVisualise(localiser, tracker)

    @property
    def _gt_poses(self) -> List[Dict]:
        poses = []
        for record in self._recording:
            if "game_pose" in record:
                poses.append(record["game_pose"])
        return poses

    def run(self):
        for record in self._recording:
            if "control_command" in record:
                # Advance particles
                time_for_step = self._step_particles(record)
                self._visualiser.update_particles()
            if "centre" in record:
                # Score Particles
                time_for_step = self._score_particles(record)
                self._visualiser.update_detections(record)
            self._tracker.update(time_for_step)

    def _step_particles(self, record: Dict):
        start_time = time.time()
        self._localiser.step_particles(record)
        return time.time() - start_time

    def _score_particles(self, observation: Dict):
        start_time = time.time()
        self._localiser.score_particles(observation)
        return time.time() - start_time
