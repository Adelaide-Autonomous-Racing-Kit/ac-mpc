from __future__ import annotations
import time
import multiprocessing as mp
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger
from utils.kdtree import KDTree
from ace.steering import SteeringGeometry

from perception.shared_memory import SharedPoints
from utils.fast_distributions import FastNormalDistribution


class Localiser:
    def __init__(self, cfg: Dict, perceiver: PerceptionProcess):
        self._previous_timestamp = time.time()
        self._localiser = LocalisationProcess(cfg, perceiver)
        self._localiser.start()

    @property
    def _dt(self) -> float:
        self._current_timestamp = time.time()
        dt = self._current_timestamp - self._previous_timestamp
        self._previous_timestamp = self._current_timestamp
        return dt

    def step(self, control_input: Tuple[float]):
        """
        Move the particles based on the control input
            Applies noise to the control input to model uncertainty
            in the movements
            Control input: (steering, acceleration, velocity)
        """
        tyre_angle = -self._localiser.angle_from_control(control_input[0])
        control_input = (tyre_angle, *control_input[1:])
        delta, velocity = self._add_noise_to_control(control_input)
        x_dot = self._calculate_x_dot(delta, velocity)
        self._advance_particles(x_dot)

    def _add_noise_to_control(self, control_input: Tuple[float]) -> Tuple[float]:
        delta = control_input[0] + self._sample_yaw_noise()
        velocity = np.abs(control_input[2] + self._sample_velocity_noise())
        return delta, velocity

    def _sample_yaw_noise(self) -> np.array:
        mu, sigma, n = 0, self.control_noise["yaw"], len(self.particles["state"])
        return self._sample_gaussian_noise(mu, sigma, n)

    def _sample_velocity_noise(self) -> np.array:
        mu, sigma, n = 0, self.control_noise["velocity"], len(self.particles["state"])
        return self._sample_gaussian_noise(mu, sigma, n)

    @staticmethod
    def _sample_gaussian_noise(mu: float, sigma: float, n_samples) -> np.array:
        return np.random.normal(mu, sigma, n_samples)

    def _calculate_x_dot(self, delta: float, velocity: float) -> np.array:
        wheel_base = self._localiser.wheel_base
        phi = self.particles["state"][:, 2]
        x_dot = np.zeros_like(self.particles["state"])
        ## w.r.t. center
        # beta = np.arctan(0.5 * np.tan(delta))
        # x_dot[:, 0] = velocity * np.cos(phi + beta)
        # x_dot[:, 1] = velocity * np.sin(phi + beta)
        # x_dot[:, 2] = velocity * np.sin(beta) / (self.wheel_base / 2)
        # w.r.t. the back axle
        x_dot[:, 0] = velocity * np.cos(phi)
        x_dot[:, 1] = velocity * np.sin(phi)
        x_dot[:, 2] = velocity * np.tan(delta) / wheel_base
        return x_dot

    def _advance_particles(self, x_dot: np.array):
        with self._localiser.particle_state_lock:
            particle_states = self._localiser.particle_states
            particle_states += x_dot * self._dt
            self._localiser.particle_states = particle_states

    @property
    def estimated_position(self) -> np.array:
        return self._estimate_location()

    @property
    def estimated_map_index(self) -> int:
        estimated_location = self._estimate_location()
        _, centre_idx = self.centre_track.query(estimated_location[:2])
        return centre_idx

    def _estimate_location(self) -> np.array:
        scores = self._localiser.particle_scores.reshape(-1, 1)
        locations = self._localiser.particle_states[:, :3]
        estimated_location = sum(locations * scores) / sum(scores)
        if any(np.isnan(estimated_location)):
            n_particles = scores.shape
            scores = np.ones(n_particles) / n_particles[0]
            estimated_location = sum(locations * scores) / sum(scores)
        return estimated_location


class LocalisationProcess(mp.Process):
    def __init__(self, cfg: Dict, perceiver: PerceptionProcess):
        super().__init__()
        self._perceiver = perceiver
        self.__setup(cfg)

    @property
    def wheel_base(self) -> float:
        return self._steering_geometry.vehicle_data.wheelbase

    def angle_from_control(self, steering_input: float) -> float:
        return self._steering_geometry.steering_angle(steering_input)

    @property
    def particle_states(self) -> np.array:
        return self._shared_particle_states.points

    @particle_states.setter
    def particle_states(self, states: np.array):
        self._shared_particle_states.points = states

    @property
    def particle_scores(self) -> np.array:
        return self._shared_particle_scores.points

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

    def run(self):
        while self.is_running:
            if not self._perceiver.is_tracklimits_stale:
                tracks = self._perceiver.tracklimits
                self._score_particles([tracks["left"], tracks["right"]])
            continue

    def _score_particles(self, observation: List[np.array]):
        observation = self.downsample_observation(observation)
        particles = self._update_particles(observation)
        is_reset = self._resample(particles)
        if is_reset:
            logger.warning("Localisation reset")
            particles = self._update_particles(observation)
        self._set_particles(particles)

    def _downsample_observations(self, observations: List[np.array]) -> List[np.array]:
        downsampled_observations = []
        for observation in observations:
            downsampled_observations.append(self._downsample_observation(observation))
        return downsampled_observations

    def _downsample_observation(self, observation: np.array) -> np.array:
        distance = np.mean(np.linalg.norm(observation[1:] - observation[:-1], axis=1))
        downsample = distance / self._average_distance_between_map_points
        n_points = len(observation) * (downsample)
        mask = np.zeros(len(observation), dtype=np.bool8)
        indices = np.linspace(0, len(observation) - 1, int(n_points), dtype=np.uint16)
        mask[indices] = True
        return observation[mask]

    def _update_particles(self, observations: List[np.array]):
        particles = {}
        particles["states"] = self.particle_states
        self._update_particle_closest_points(particles)
        self._update_particle_heading_offsets(particles)
        self._update_particle_error(observations, particles)
        return particles

    def _update_particle_closest_points(self, particles: Dict):
        particle_locations = particles["states"]
        centre_offsets, track_indices = self.find_closest_points_to_particles(
            particle_locations
        )
        particles["track_indices"] = track_indices
        particles["centreline_idx"] = track_indices[:, 0]
        particles["minimum_offset"] = centre_offsets

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
        heading_offset = track_heading - particles["state"][:, 2]
        return (heading_offset + np.pi) % (2 * np.pi) - np.pi

    def _update_particle_error(self, observation: List[np.array], particles: Dict):
        score, error = self._calculate_particle_error(observation, particles)
        particles["observation_error"] = error
        particles["score"] = score

    def _calculate_particle_error(self, observation: List[np.array], particles: Dict):
        observations = self._process_observation(observation, particles)
        track_limits = self._process_track_limits(observation, particles)
        return self._calculate_error(observations, track_limits)

    def _process_observation(
        self,
        observation: List[np.array],
        particles: Dict,
    ) -> np.array:
        # observation[0][:, 0] = np.clip(observation[0][:, 0], -200, 200)
        # observation[0][:, 1] = np.clip(observation[0][:, 1], 0, 50)
        # observation[1][:, 0] = np.clip(observation[1][:, 0], -200, 200)
        # observation[1][:, 1] = np.clip(observation[1][:, 1], 0, 50)
        observation[0] = observation[0][observation[0][:, 1] < 50]
        observation[1] = observation[1][observation[1][:, 1] < 50]
        map_rot = self._calculate_map_rotation(particles)
        observations = np.concatenate(observation)
        n_points = observations.shape[0]
        n_particles = len(particles["state"])
        # Repeat observation for each particle the reshape for particles x points x 2 (x,y)
        observations = np.tile(observations, (n_particles, 1))
        observations = observations.reshape(n_particles, -1, 2).transpose(0, 2, 1)
        observations = np.matmul(map_rot, observations)
        observations = observations.transpose(0, 2, 1)
        particle_states = np.tile(particles["state"][:, :2], (1, n_points)).reshape(
            n_particles, n_points, 2
        )
        return np.add(observations, particle_states)

    def _calculate_map_rotation(self, particles: Dict) -> np.array:
        angle = -particles["state"][:, 2] + np.pi / 2
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
        observation: List[np.array],
        particles: Dict,
    ) -> np.array:
        left_track_points = self._get_left_track_limits(observation, particles)
        right_track_points = self._get_right_track_limits(observation, particles)
        return np.concatenate([left_track_points, right_track_points], axis=1)

    def get_left_track_limits(
        self,
        observation: np.array,
        particles: Dict,
    ) -> np.array:
        closest = particles["track_indices"][:, 1]
        return self._get_track_limit(closest, observation[0], self.left_track)

    def get_right_track_limits(
        self,
        observation: np.array,
        particles: Dict,
    ) -> np.array:
        closest = particles["track_indices"][:, 2]
        return self._get_track_limit(closest, observation[1], self.right_track)

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
        observations: np.array,
        track_limits: np.array,
    ) -> Tuple[float]:
        errors = np.sqrt(np.sum(np.subtract(observations, track_limits) ** 2, axis=2))
        average_track_errors = np.mean(errors, axis=1)
        score = self.pdf(np.copy(average_track_errors)) / self.scale
        return score, average_track_errors

    def _resample(self, particles: Dict) -> bool:
        """
        This function takes particles and randomly samples them
        in proportion to their scores, replacing ones lower than thresholds
        """
        self._remove_invalid_particles(particles)
        if self._is_too_few_particles(particles):
            # Lost too many particles, reset
            self._sampling_strategy(particles)
            return True
        self._add_new_particles(particles)
        return False

    def _remove_invalid_particles(self, particles: Dict):
        valid_particle_mask = self._get_valid_particle_mask(particles)
        for key, val in particles.items():
            particles[key] = val[valid_particle_mask]

    def _get_valid_particle_mask(self, particles: Dict) -> np.array:
        sample_mask = (
            # (self.particles["heading_offset"] < self.thresholds["rotation"])
            (particles["minimum_offset"] < self.thresholds["offset"])
            & (particles["observation_error"] < self.thresholds["track_limit"])
        )
        return sample_mask

    def _is_too_few_particles(self, particles: Dict) -> bool:
        n_particles = particles["state"].shape[0]
        return n_particles < self.thresholds["minimum_particles"]

    def _sampling_strategy(self):
        particle_start_point_idx = np.linspace(
            0, len(self.centre_track) - 3, self.max_number_of_particles
        ).astype(np.int16)
        x1 = self.centre_track[particle_start_point_idx, 0]
        y1 = self.centre_track[particle_start_point_idx, 1]
        x2 = self.centre_track[particle_start_point_idx + 1, 0]
        y2 = self.centre_track[particle_start_point_idx + 1, 1]
        yaws = np.arctan2(y2 - y1, x2 - x1)
        samples = np.vstack((x1, y1, yaws)).T
        scores = np.ones(self.max_number_of_particles)
        scores /= np.sum(scores)

        self.localised = False
        with self.particle_state_lock:
            self.particle_scores = scores
            self.particle_states = samples

    def _add_new_particles(self, particles: Dict):
        n_new_particles = self._get_n_new_particles_to_create(particles)
        noise = self._generate_sampling_noise(n_new_particles)
        indices = self._sample_current_particle_indices(n_new_particles, particles)
        new_particle_states = particles["state"][indices] + noise
        self._add_new_particle_states(new_particle_states, particles)
        self._add_new_particle_scores(particles, indices)

    def _get_n_new_particles_to_create(self, particles: Dict) -> int:
        n_particles = particles["state"].shape[0]
        n_desired_particles = self._get_desired_n_particles()
        return max(0, n_desired_particles - n_particles)

    def _get_desired_n_particles(self) -> int:
        if self.localised:
            return self._n_converged_particles
        return self._max_n_particles

    def _generate_sampling_noise(self, n_samples: int) -> np.array:
        x_noise = self._sample_particle_x_noise(n_samples)
        y_noise = self._sample_particle_y_noise(n_samples)
        yaw_noise = self._sample_particle_yaw_noise(n_samples)
        return np.array([x_noise, y_noise, yaw_noise]).T

    def _sample_particle_x_noise(self, n_samples: int) -> np.array:
        mu, sigma = 0, self.sampling_noise["x"]
        return self._sample_gaussian_noise(mu, sigma, n_samples)

    def _sample_particle_y_noise(self, n_samples: int) -> np.array:
        mu, sigma = 0, self.sampling_noise["y"]
        return self._sample_gaussian_noise(mu, sigma, n_samples)

    def _sample_particle_yaw_noise(self, n_samples: int) -> np.array:
        mu, sigma = 0, self.sampling_noise["yaw"]
        return self._sample_gaussian_noise(mu, sigma, n_samples)

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

    def _add_new_particle_states(self, new_states: np.array, particles: Dict):
        updated_states = np.concatenate((particles["state"], new_states), axis=0)
        particles["state"] = updated_states

    def _add_new_particle_scores(self, particles: Dict, particle_indices: np.array):
        new_scores = particles["score"][particle_indices]
        updated_scores = np.concatenate((particles["score"], new_scores), axis=0)
        particles["score"] = updated_scores

    def _set_particles(self, particles: Dict):
        with self.particle_state_lock:
            self.particle_states = particles["states"]
            self.particle_scores = particles["scores"]

    def __setup(self, cfg: Dict):
        self.__setup_config(cfg)
        self.__setup_shared_memory()
        self.__setup_localiser(cfg)

    def __setup_config(self, cfg: Dict):
        self._steering_geometry = SteeringGeometry(cfg["vehicle"]["data_path"])
        self._map_path = self.cfg["mapping"]["map_path"]
        self._unpack_noise_config(cfg)
        self._unpack_convergence_config(cfg)
        self._unpack_particle_config(cfg)

    def _unpack_noise_config(self, cfg: Dict):
        self._sampling_noise = cfg["sampling_noise"]
        self._sampling_noise["yaw"] *= np.pi / 180
        self._control_noise = cfg["control_noise"]
        self._control_noise["yaw"] *= np.pi / 180

    def _unpack_convergence_config(self, cfg: Dict):
        self._convergence_criteria = cfg["convergence_criteria"]
        self._thresholds = cfg["thresholds"]
        self._thresholds["rotation"] *= np.pi / 180

    def _unpack_particle_config(self, cfg: Dict):
        self._score_distribution_mean = cfg["score_distribution"]["mean"]
        self._score_distribution_sigma = cfg["score_distribution"]["sigma"]
        self._max_n_particles = cfg["n_particles"]
        self._n_converged_particles = cfg["n_converged_particles"]

    def __setup_shared_memory(self):
        self.particle_state_lock = mp.Lock()
        self._shared_particle_scores = SharedPoints(self._max_n_particles, 0)
        self._shared_particle_states = SharedPoints(self._max_n_particles, 3)
        self._is_running = mp.Value("i", True)

    def __setup_localiser(self):
        self._load_map()
        self._initialise_score_distribution()

    def _load_map(self) -> Dict:
        map_dict = np.load(self._map_path, allow_pickle=True).item()
        self.centre_track = KDTree(map_dict["centre"])
        self.left_track = KDTree(map_dict["left"])
        self.right_track = KDTree(map_dict["right"])
        self._set_average_distance_between_map_points(map_dict["centre"])

    def _set_average_distance_between_map_points(self, centre: np.array):
        distances = np.linalg.norm(centre[1:] - centre[:-1], axis=1)
        self._average_distance_between_map_points = np.mean(distances)

    def _initialise_score_distribution(self):
        mean, sigma = self._score_distribution_mean, self._score_distribution_mean
        self._distribution = FastNormalDistribution(mean, sigma)
        self._pdf = self.distribution.pdf
        self._scale = np.max(self.pdf(np.linspace(-10, 10, 100)))
