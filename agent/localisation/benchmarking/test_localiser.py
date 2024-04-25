from typing import Dict

import numpy as np

from localisation.localiser import Localiser, LocalisationProcess
from utils.fast_distributions import FastNormalDistribution


class TestLocaliser(Localiser):
    def __init__(self, cfg: Dict):
        self._localiser = TestLocalisationProcess(cfg)
        self._last_timestamp = None

    @property
    def particle_states(self) -> np.array:
        return self._localiser.particle_states

    @property
    def particle_scores(self) -> np.array:
        return self._localiser.particle_scores

    @property
    def particle_errors(self) -> np.array:
        return self._localiser.particles["observation_error"]

    @property
    def pdf(self) -> FastNormalDistribution:
        return self._localiser._pdf

    @property
    def scale(self) -> float:
        return self._localiser._scale

    def score_particles(self, observation: Dict):
        self._localiser._score_particles(observation)

    @property
    def _dt(self) -> float:
        return self.dt

    def step_particles(self, control_record: Dict):
        self._update_control_timestamp(control_record["time"])
        self.step(control_record["control_command"])

    def _update_control_timestamp(self, timestamp: float):
        if self._last_timestamp is None:
            self._last_timestamp = timestamp
        self.dt = timestamp - self._last_timestamp
        self._last_timestamp = timestamp


class TestLocalisationProcess(LocalisationProcess):
    def __init__(self, cfg: Dict):
        self._setup(cfg)

    def _score_particles(self, observation: Dict):
        observation = self._downsample_observations(observation)
        particles = self._update_particles(observation)
        self._resample_particles(particles)
        self._update_is_converged_flag()
        self.particles = particles
