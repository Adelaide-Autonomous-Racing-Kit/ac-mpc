import time
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from localisation.benchmarking.tracker import LocalisationTracker
from localisation.benchmarking.utils import LocalisationRecording
from localisation.benchmarking.visualisation import LocalisationVisualiser
from localisation.benchmarking.test_localiser import TestLocaliser


class BenchmarkLocalisation:
    def __init__(self, cfg: Dict):
        self.__setup(cfg)

    def __setup(self, cfg: Dict):
        self._unpack_config(cfg)
        self._seed_numpy()
        self._recording = LocalisationRecording(self._data_path)
        self._setup_particle_filter()
        self._setup_tracker(self._localiser)
        self._setup_visualiser(self._localiser, self._tracker)
        self._n_observations = 0

    def _unpack_config(self, cfg: Dict):
        self._cfg = cfg
        self._data_path = cfg["data_path"]
        self._n_observations_between_plots = cfg["n_observations_between_plots"]
        self._seed = cfg["seed"]

    def _seed_numpy(self):
        np.random.seed(self._seed)

    def _setup_particle_filter(self) -> TestLocaliser:
        self._localiser = TestLocaliser(self._cfg)

    def _setup_tracker(self, localiser: TestLocaliser) -> LocalisationTracker:
        self._tracker = LocalisationTracker(localiser, self._gt_poses)

    def _setup_visualiser(
        self,
        localiser: TestLocaliser,
        tracker: LocalisationTracker,
    ):
        self._visualiser = LocalisationVisualiser(localiser, tracker)

    @property
    def _gt_poses(self) -> List[Dict]:
        poses = []
        for record in self._recording:
            if "game_pose" in record:
                poses.append(record["game_pose"])
        return poses

    def run(self):
        for record in tqdm(self._recording):
            if "control_command" in record:
                time_for_step = self._step_particles(record)
                #  self._visualiser.update_particles()
                self._tracker.update_step(time_for_step)
            if "tracklimits" in record:
                observation = record["tracklimits"]
                time_for_step = self._score_particles(observation)
                if self._n_observations % self._n_observations_between_plots == 0:
                    self._visualiser.update_detections(observation)
                    pass
                self._tracker.update_observation(time_for_step)
                self._n_observations += 1

    def _step_particles(self, record: Dict):
        start_time = time.time()
        self._localiser.step_particles(record)
        return time.time() - start_time

    def _score_particles(self, observation: Dict):
        start_time = time.time()
        self._localiser.score_particles(observation)
        return time.time() - start_time
