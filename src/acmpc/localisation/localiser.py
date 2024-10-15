from __future__ import annotations

import multiprocessing as mp
import signal
import time
from typing import Dict, List, Tuple

from ace.steering import SteeringGeometry
from aci.utils.system_monitor import SystemMonitor, track_runtime
from loguru import logger
import numpy as np
from perception.shared_memory import SharedPoints
from utils import load
from utils.fast_distributions import FastNormalDistribution
from utils.kdtree import KDTree

Localisation_Monitor = SystemMonitor(300)


class Localiser:
    def __init__(self, cfg: Dict, perceiver: PerceptionProcess):
        self._previous_timestamp = time.time()
        self._localiser = LocalisationProcess(cfg, perceiver)

    @property
    def n_particles(self) -> int:
        return 0

    @property
    def _dt(self) -> float:
        self._current_timestamp = time.time()
        dt = self._current_timestamp - self._previous_timestamp
        self._previous_timestamp = self._current_timestamp
        return dt

    def start(self):
        self._localiser.start()

    def shutdown(self):
        self._localiser.is_running = False

    def step(self, control_input: Tuple[float]):
        """
        Move the particles based on the control input
            Applies noise to the control input to model uncertainty
            in the movements
            Control input: (steering, acceleration, velocity)
        """
        tyre_angle = -self._localiser.angle_from_control(control_input[0])
        control_input = (tyre_angle, *control_input[1:])
        self._advance_particles(control_input)

    def _add_noise_to_control(
        self,
        control_input: Tuple[float],
        n_particles: int,
    ) -> Tuple[float]:
        delta = control_input[0] + self._localiser.sample_control_noise_yaw(n_particles)
        velocity = np.abs(
            control_input[2]
            + self._localiser.sample_control_noise_velocity(n_particles)
        )
        return delta, velocity

    def _advance_particles(self, control_input: Tuple[float]):
        states = self._localiser._default_state_array()
        with self._localiser.particle_lock:
            particle_states = self._localiser.particle_states
            n_particles = particle_states.shape[0]
            delta, velocity = self._add_noise_to_control(control_input, n_particles)
            x_dot = self._calculate_x_dot(delta, particle_states, velocity)
            particle_states += x_dot * self._dt
            states[:n_particles] = particle_states
            self._localiser.particle_states = states

    def _calculate_x_dot(
        self,
        delta: float,
        particle_states: np.array,
        velocity: float,
    ) -> np.array:
        wheel_base = self._localiser.wheel_base
        phi = particle_states[:, 2]
        x_dot = np.zeros_like(particle_states)
        # w.r.t. center
        # beta = np.arctan(0.5 * np.tan(delta))
        # x_dot[:, 0] = velocity * np.cos(phi + beta)
        # x_dot[:, 1] = velocity * np.sin(phi + beta)
        # x_dot[:, 2] = velocity * np.sin(beta) / (self.wheel_base / 2)
        # w.r.t. the back axle
        x_dot[:, 0] = velocity * np.cos(phi)
        x_dot[:, 1] = velocity * np.sin(phi)
        x_dot[:, 2] = velocity * np.tan(delta) / wheel_base
        return x_dot

    @property
    def is_localised(self) -> bool:
        return self._localiser.is_converged

    @property
    def estimated_position(self) -> np.array:
        return self._localiser.estimated_location

    @property
    def estimated_map_index(self) -> int:
        estimated_position = self.estimated_position
        _, i_centre = self._localiser.centre_track.query(estimated_position[:2])
        return i_centre

    @property
    def visualisation_estimated_position(self) -> Tuple:
        estimated_position = self.estimated_position
        _, i_centre = self._localiser.centre_track.query(estimated_position[:2])
        _, i_left = self._localiser.left_track.query(estimated_position[:2])
        _, i_right = self._localiser.right_track.query(estimated_position[:2])
        return estimated_position, i_centre, i_left, i_right

    @property
    def centre_track(self) -> KDTree:
        return self._localiser.centre_track

    @property
    def left_track(self) -> KDTree:
        return self._localiser.left_track

    @property
    def right_track(self) -> KDTree:
        return self._localiser.right_track


class LocalisationProcess(mp.Process):
    def __init__(self, cfg: Dict, perceiver: PerceptionProcess):
        super().__init__()
        self._perceiver = perceiver
        self._setup(cfg)

    @property
    def estimated_location(self) -> np.array:
        with self.particle_lock:
            scores = self.particle_scores
            states = self.particle_states
        return self._estimate_location(scores, states)

    @property
    def wheel_base(self) -> float:
        return self._steering_geometry.vehicle_data.wheelbase

    def angle_from_control(self, steering_input: float) -> float:
        return self._steering_geometry.steering_angle(steering_input)

    @property
    def particle_states(self) -> np.array:
        scores = self._shared_particle_scores.points
        particles = self._shared_particle_states.points
        return particles[scores > 0]

    @particle_states.setter
    def particle_states(self, states: np.array):
        self._shared_particle_states.points = states

    @property
    def particle_scores(self) -> np.array:
        scores = self._shared_particle_scores.points
        return scores[scores > 0]

    @particle_scores.setter
    def particle_scores(self, scores: np.array):
        self._shared_particle_scores.points = scores

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

    @property
    def is_converged(self) -> bool:
        """
        Checks if the filter has converged

        :return: True if the filter has converged, false if it is not
        :rtype: bool
        """
        with self._is_converged.get_lock():
            is_converged = self._is_converged.value
        return is_converged

    @is_converged.setter
    def is_converged(self, is_converged: bool):
        """
        Sets if the filter has converged

        :is_converged: True if the filter has converged, false if it is not
        :type is_converged: bool
        """
        with self._is_converged.get_lock():
            self._is_converged.value = is_converged
        if is_converged:
            self._is_previously_converged = True

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while self.is_running:
            if self._perceiver.is_tracklimits_stale:
                continue
            if self._is_collecting_localisation_data:
                self._cache_observation(self._perceiver.visualisation_tracks)
            self._score_particles(self._perceiver.tracklimits)
        self._maybe_save_observations()
        # Localisation_Monitor.maybe_log_function_itterations_per_second()

    def _cache_observation(self, observation: Dict):
        self._tracklimit_observation = observation

    @track_runtime(Localisation_Monitor)
    def _score_particles(self, observation: Dict):
        observation = self._downsample_observations(observation)
        particles = self._update_particles(observation)
        self._resample_particles(particles)
        self._update_is_converged_flag()

    def _downsample_observations(self, observations: List[np.array]) -> List[np.array]:
        track_left = self._downsample_observation(observations["left"])
        track_right = self._downsample_observation(observations["right"])
        return [track_left, track_right]

    def _downsample_observation(self, observation: np.array) -> np.array:
        distance = np.mean(np.linalg.norm(observation[1:] - observation[:-1], axis=1))
        downsample = distance / self._average_distance_between_map_points
        n_points = len(observation) * (downsample)
        mask = np.zeros(len(observation), dtype=np.bool_)
        indices = np.linspace(0, len(observation) - 1, int(n_points), dtype=np.uint16)
        mask[indices] = True
        return observation[mask]

    def _update_particles(self, observations: List[np.array]):
        particles = {}
        if self._is_collecting_localisation_data:
            self._add_observation()
        particles["states"] = self.particle_states
        self._update_particle_closest_points(particles)
        self._update_particle_heading_offsets(particles)
        self._update_particle_error(observations, particles)
        self._update_particle_scores(particles)
        return particles

    def _add_observation(self):
        self._observations[self._observation_count] = {
            "tracklimits": self._tracklimit_observation,
            "time": time.time(),
        }
        self._observation_count += 1

    def _update_particle_closest_points(self, particles: Dict):
        particle_locations = particles["states"][:, :2]
        offsets, track_indices = self._find_closest_points_to_particles(
            particle_locations
        )
        particles["track_indices"] = track_indices
        particles["centreline_idx"] = track_indices[:, 0]
        particles["minimum_offset"] = offsets

    def _find_closest_points_to_particles(
        self,
        particle_locations: np.array,
    ) -> List[np.array]:
        _, right_idx = self.right_track.query(particle_locations)
        _, left_idx = self.left_track.query(particle_locations)
        centre_offsets, centre_idx = self.centre_track.query(particle_locations)
        return centre_offsets, np.array([centre_idx, left_idx, right_idx]).T

    def _update_particle_heading_offsets(self, particles: Dict):
        centreline_idx = particles["centreline_idx"]
        centreline_points = self._get_track_points_by_index(centreline_idx)
        next_points = self._get_track_points_by_index(centreline_idx + 1)
        heading_offset = self._calculate_heading_offset(
            centreline_points,
            particles,
            next_points,
        )
        # logger.info(f"Headings {heading_offset * 180 / np.pi}")
        particles["heading_offset"] = heading_offset

    def _get_track_points_by_index(self, indices: np.array) -> np.array:
        return self.centre_track[np.mod(indices, len(self.centre_track) - 1)]

    def _calculate_heading_offset(
        self,
        centreline_points: np.array,
        particles: Dict,
        next_points: np.array,
    ) -> np.array:
        track_heading = np.arctan2(
            next_points[:, 1] - centreline_points[:, 1],
            next_points[:, 0] - centreline_points[:, 0],
        )
        # logger.debug(f"Track Heading {track_heading * 180 / np.pi}")
        heading_offset = track_heading - particles["states"][:, 2]
        return np.abs((heading_offset + np.pi) % (2 * np.pi) - np.pi)

    def _update_particle_error(self, observations: List[np.array], particles: Dict):
        score, error = self._calculate_particle_error(observations, particles)
        particles["observation_error"] = error
        particles["score"] = score

    def _calculate_particle_error(self, observations: List[np.array], particles: Dict):
        observation = self._process_observations(observations, particles)
        track_limits = self._process_track_limits(observations, particles)
        return self._calculate_error(observation, track_limits)

    def _process_observations(
        self,
        observations: List[np.array],
        particles: Dict,
    ) -> np.array:
        # observation[0][:, 0] = np.clip(observation[0][:, 0], -200, 200)
        # observation[0][:, 1] = np.clip(observation[0][:, 1], 0, 50)
        # observation[1][:, 0] = np.clip(observation[1][:, 0], -200, 200)
        # observation[1][:, 1] = np.clip(observation[1][:, 1], 0, 50)
        observations[0] = observations[0][observations[0][:, 1] < 50]
        observations[1] = observations[1][observations[1][:, 1] < 50]
        map_rot = self._calculate_map_rotation(particles)
        observation = np.concatenate(observations)
        n_points = observation.shape[0]
        n_particles = len(particles["states"])
        # Repeat observation for each particle the reshape for particles x points x 2 (x,y)
        observation = np.tile(observation, (n_particles, 1))
        observation = observation.reshape(n_particles, -1, 2).transpose(0, 2, 1)
        observation = np.matmul(map_rot, observation)
        observation = observation.transpose(0, 2, 1)
        particle_states = np.tile(particles["states"][:, :2], (1, n_points)).reshape(
            n_particles, n_points, 2
        )
        return np.add(observation, particle_states)

    def _calculate_map_rotation(self, particles: Dict) -> np.array:
        angle = -particles["states"][:, 2] + np.pi / 2
        map_rot = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        # transpose (inverse) since rotation is from observation to particle now
        return map_rot.transpose(2, 1, 0)

    def _process_track_limits(
        self,
        observations: List[np.array],
        particles: Dict,
    ) -> np.array:
        left_track_points = self._get_left_track_limits(observations, particles)
        right_track_points = self._get_right_track_limits(observations, particles)
        return np.concatenate([left_track_points, right_track_points], axis=1)

    def _get_left_track_limits(
        self,
        observations: List[np.array],
        particles: Dict,
    ) -> np.array:
        closest = particles["track_indices"][:, 1]
        return self._get_track_limit(closest, observations[0], self.left_track)

    def _get_right_track_limits(
        self,
        observations: List[np.array],
        particles: Dict,
    ) -> np.array:
        closest = particles["track_indices"][:, 2]
        return self._get_track_limit(closest, observations[1], self.right_track)

    def _get_track_limit(
        self,
        closest: np.array,
        observation: np.array,
        track: KDTree,
    ) -> np.array:
        n_points = observation.shape[0]
        track_idxs = np.linspace(closest, closest + n_points, n_points, dtype=np.uint16)
        track_idxs = np.mod(track_idxs, len(track)).T
        return track[track_idxs]

    def _calculate_error(
        self,
        observation: np.array,
        track_limits: np.array,
    ) -> Tuple[float]:
        errors = np.linalg.norm(observation - track_limits, axis=2)
        average_track_errors = np.mean(errors, axis=1)
        score = self._pdf(np.copy(average_track_errors)) / self._scale
        return score, average_track_errors

    def _update_particle_scores(self, particles: Dict):
        scores = -1.0 * self._default_score_array()
        with self.particle_lock:
            particle_scores = self.particle_scores
            n_particles = particle_scores.shape[0]
            scores[:n_particles] = particles["score"]
            self.particle_scores = scores

    def _resample_particles(self, particles: Dict) -> bool:
        """
        This function takes particles and randomly samples them
        in proportion to their scores, replacing ones lower than thresholds
        """
        self._remove_invalid_particles(particles)
        if self._is_too_few_particles(particles):
            self._reset_filter()
            if self._is_previously_converged:
                logger.warning("Particle filter reset")
            return
        self._add_new_particles(particles)

    def _remove_invalid_particles(self, particles: Dict):
        valid_particle_mask = self._get_valid_particle_mask(particles)
        n_valid_particles = sum(valid_particle_mask)
        scores = -1.0 * self._default_score_array()
        states = self._default_state_array()
        with self.particle_lock:
            if n_valid_particles > 0:
                scores[:n_valid_particles] = self.particle_scores[valid_particle_mask]
                states[:n_valid_particles] = self.particle_states[valid_particle_mask]
            self.particle_scores = scores
            self.particle_states = states
        for key, val in particles.items():
            particles[key] = val[valid_particle_mask]

    def _default_score_array(self) -> np.array:
        return np.ones(self._max_n_particles, dtype=np.float32)

    def _default_state_array(self) -> np.array:
        return np.zeros((self._max_n_particles, 3), dtype=np.float32)

    def _get_valid_particle_mask(self, particles: Dict) -> np.array:
        sample_mask = (
            (particles["heading_offset"] < self._threshold_rotation)
            & (particles["minimum_offset"] < self._threshold_offset)
            & (particles["observation_error"] < self._threshold_error)
        )
        # logger.info(particles["observation_error"])
        # logger.info(particles["minimum_offset"])
        # logger.info(particles["heading_offset"] * 180 / np.pi)
        return sample_mask

    def _is_too_few_particles(self, particles: Dict) -> bool:
        n_particles = particles["states"].shape[0]
        return n_particles < self._threshold_n_particles

    def _reset_filter(self):
        particle_start_point_idx = np.linspace(
            0, len(self.centre_track) - 3, self._max_n_particles
        ).astype(np.int16)
        x1 = self.centre_track[particle_start_point_idx, 0]
        y1 = self.centre_track[particle_start_point_idx, 1]
        x2 = self.centre_track[particle_start_point_idx + 1, 0]
        y2 = self.centre_track[particle_start_point_idx + 1, 1]
        yaws = np.arctan2(y2 - y1, x2 - x1)
        states = np.vstack((x1, y1, yaws)).T
        scores = self._default_score_array()
        scores /= np.sum(scores)

        self.is_converged = False
        with self.particle_lock:
            self.particle_scores = scores
            self.particle_states = states

    def _add_new_particles(self, particles: Dict):
        n_new_particles = self._get_n_new_particles_to_create(particles)
        noise = self._generate_sampling_noise(n_new_particles)
        indices = self._sample_current_particle_indices(n_new_particles, particles)
        new_particle_states = self.particle_states[indices] + noise
        new_particle_scores = self.particle_scores[indices]
        with self.particle_lock:
            self._add_new_particle_states(new_particle_states)
            self._add_new_particle_scores(new_particle_scores)

    def _get_n_new_particles_to_create(self, particles: Dict) -> int:
        n_particles = particles["states"].shape[0]
        n_desired_particles = self._get_desired_n_particles()
        return max(0, n_desired_particles - n_particles)

    def _get_desired_n_particles(self) -> int:
        if self.is_converged:
            return self._n_converged_particles
        return self._max_n_particles

    def _generate_sampling_noise(self, n_samples: int) -> np.array:
        x_noise = self._sample_particle_x_noise(n_samples)
        y_noise = self._sample_particle_y_noise(n_samples)
        yaw_noise = self._sample_particle_yaw_noise(n_samples)
        return np.array([x_noise, y_noise, yaw_noise]).T

    def _sample_particle_x_noise(self, n_samples: int) -> np.array:
        mu, sigma = 0, self._sampling_noise_x
        return self._sample_gaussian_noise(mu, sigma, n_samples)

    def _sample_particle_y_noise(self, n_samples: int) -> np.array:
        mu, sigma = 0, self._sampling_noise_y
        return self._sample_gaussian_noise(mu, sigma, n_samples)

    def _sample_particle_yaw_noise(self, n_samples: int) -> np.array:
        mu, sigma = 0, self._sampling_noise_yaw
        return self._sample_gaussian_noise(mu, sigma, n_samples)

    def sample_control_noise_yaw(self, n_particles: int) -> np.array:
        mu, sigma, n = 0, self._control_noise_yaw, n_particles
        return self._sample_gaussian_noise(mu, sigma, n)

    def sample_control_noise_velocity(self, n_particles: int) -> np.array:
        mu, sigma, n = 0, self._control_noise_velocity, n_particles
        return self._sample_gaussian_noise(mu, sigma, n)

    @staticmethod
    def _sample_gaussian_noise(mu: float, sigma: float, n_samples) -> np.array:
        return np.random.normal(mu, sigma, n_samples)

    def _sample_current_particle_indices(
        self,
        n_samples: int,
        particles: Dict,
    ) -> np.array:
        scores = particles["score"]
        weights = scores / np.sum(scores)
        if any(np.isnan(weights)):
            weights = np.ones(scores.shape) / scores.shape[0]
        return np.random.choice(scores.shape[0], size=n_samples, p=weights)

    def _add_new_particle_states(self, new_states: np.array):
        states = self._default_state_array()
        updated_states = np.concatenate((self.particle_states, new_states), axis=0)
        n_particles = updated_states.shape[0]
        states[:n_particles] = updated_states
        self.particle_states = states

    def _add_new_particle_scores(self, new_scores: np.array):
        scores = -1.0 * self._default_score_array()
        updated_scores = np.concatenate((self.particle_scores, new_scores), axis=0)
        n_particles = updated_scores.shape[0]
        scores[:n_particles] = updated_scores
        self.particle_scores = scores

    def _update_is_converged_flag(self):
        with self.particle_lock:
            scores = self.particle_scores
            states = self.particle_states
        estimated_centre = self._estimate_location(scores, states)
        distances = np.linalg.norm(states[:, :2] - estimated_centre[:2], axis=1)
        angles = abs(states[:, 2] - estimated_centre[2])
        particles_are_close = np.max(distances) < self._convergence_distance
        headings_are_aligned = np.max(angles) < self._convergence_angle
        self.is_converged = particles_are_close and headings_are_aligned

    def _estimate_location(self, scores: np.array, states: np.array) -> np.array:
        locations, scores = states[:, :3], scores.reshape(-1, 1)
        estimated_location = sum(locations * scores) / sum(scores)
        if any(np.isnan(estimated_location)):
            n_particles = scores.shape
            scores = np.ones(n_particles) / n_particles[0]
            estimated_location = sum(locations * scores) / sum(scores)
        return estimated_location

    def _maybe_save_observations(self):
        if self._is_collecting_localisation_data:
            np.save(self._recording_path, self._observations)

    def _setup(self, cfg: Dict):
        self.__setup_config(cfg)
        self.__setup_shared_memory()
        self.__setup_localiser()

    def __setup_config(self, cfg: Dict):
        self._experiment_name = cfg["experiment_name"]
        self._steering_geometry = SteeringGeometry(cfg["vehicle"]["data_path"])
        self._map_path = cfg["mapping"]["map_path"]
        localisation_cfg = cfg["localisation"]
        self._unpack_noise_config(localisation_cfg)
        self._unpack_threshold_config(localisation_cfg)
        self._unpack_convergence_config(localisation_cfg)
        self._unpack_particle_config(localisation_cfg)
        self._unpack_recording_config(localisation_cfg)

    def _unpack_noise_config(self, cfg: Dict):
        self._sampling_noise_x = cfg["sampling_noise"]["x"]
        self._sampling_noise_y = cfg["sampling_noise"]["y"]
        self._sampling_noise_yaw = cfg["sampling_noise"]["yaw"] * np.pi / 180
        self._control_noise_velocity = cfg["control_noise"]["velocity"]
        self._control_noise_yaw = cfg["control_noise"]["yaw"] * np.pi / 180

    def _unpack_threshold_config(self, cfg: Dict):
        threshold_cfg = cfg["thresholds"]
        self._threshold_n_particles = threshold_cfg["minimum_particles"]
        self._threshold_error = threshold_cfg["track_limit"]
        self._threshold_offset = threshold_cfg["offset"]
        self._threshold_rotation = threshold_cfg["rotation"] * np.pi / 180

    def _unpack_convergence_config(self, cfg: Dict):
        self._convergence_distance = cfg["convergence_criteria"]["maximum_distance"]
        self._convergence_angle = cfg["convergence_criteria"]["maximum_angle"]

    def _unpack_particle_config(self, cfg: Dict):
        self._score_distribution_mean = cfg["score_distribution"]["mean"]
        self._score_distribution_sigma = cfg["score_distribution"]["sigma"]
        self._max_n_particles = cfg["n_particles"]
        self._n_converged_particles = cfg["n_converged_particles"]

    def _unpack_recording_config(self, cfg: Dict):
        self._is_collecting_localisation_data = cfg["collect_benchmark_observations"]
        save_path = cfg["benchmark_observations_save_location"]
        self._recording_path = f"{save_path}/{self._experiment_name}/observations.npy"
        self._observations = {}
        self._observation_count = 0

    def __setup_shared_memory(self):
        self.particle_lock = mp.Lock()
        self._shared_particle_scores = SharedPoints(self._max_n_particles, 0)
        self._shared_particle_states = SharedPoints(self._max_n_particles, 3)
        self._is_running = mp.Value("i", True)
        self._is_converged = mp.Value("i", False)

    def __setup_localiser(self):
        self._is_previously_converged = False
        self._load_map()
        self._initialise_score_distribution()
        self._reset_filter()

    def _load_map(self) -> Dict:
        map_dict = load.track_map(self._map_path)
        self.centre_track = KDTree(map_dict["centre"])
        self.left_track = KDTree(map_dict["left"])
        self.right_track = KDTree(map_dict["right"])
        self._set_average_distance_between_map_points(map_dict["centre"])

    def _set_average_distance_between_map_points(self, centre: np.array):
        distances = np.linalg.norm(centre[1:] - centre[:-1], axis=1)
        self._average_distance_between_map_points = np.mean(distances)

    def _initialise_score_distribution(self):
        mean = self._score_distribution_mean
        sigma = self._score_distribution_sigma
        self._distribution = FastNormalDistribution(mean, sigma)
        self._pdf = self._distribution.pdf
        self._scale = np.max(self._pdf(np.linspace(-10, 10, 100)))
