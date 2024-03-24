from __future__ import annotations
import multiprocessing as mp
from typing import Dict, List

import numpy as np


class Localiser:
    def __init__(self, cfg: Dict, perceiver: PerceptionProcess):
        self._localiser = LocalisationProcess(cfg, perceiver)
        self._localiser.start()

    def step(self, state: Dict):
        # Progress particles with ego state
        pass

    @property
    def estimated_position(self) -> np.array:
        # Get estimated vehicle position
        pass


class LocalisationProcess(mp.Process):
    def __init__(self, cfg: Dict, perceiver: PerceptionProcess):
        super().__init__()
        self._perceiver = perceiver
        self.__setup(cfg)

    def __setup(self, cfg: Dict):
        self.__setup_config(cfg)
        self.__setup_shared_memory()

    def __setup_config(self, cfg: Dict):
        pass

    def __setup_shared_memory(self):
        self._is_running = mp.Value("i", True)

    @property
    def is_running(self) -> bool:
        """
        Checks if the localisation process is running

        :return: True if the process is running, false if it is not
        :rtype: bool
        """
        with self._is_running.get_lock():
            is_running = self._is_running.value
        return is_running

    @is_running.setter
    def is_running(self, is_running: bool):
        """
        Sets if the localisation process is running

        :is_running: True if the process is running, false if it is not
        :type is_running: bool
        """
        with self._is_running.get_lock():
            self._is_running.value = is_running

    def run(self):
        while self.is_running:
            if not self._perceiver.is_tracklimits_stale:
                tracks = self._perceiver.tracklimits
                self._score_particles([tracks["left"], tracks["right"]])
            continue

    def _score_particles(self, observation: List[np.array]):
        # Update particle scores based on observation
        pass
