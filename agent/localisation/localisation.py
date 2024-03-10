from typing import List, Tuple

import numpy as np
from monitor.system_monitor import track_runtime
from utils.fast_distributions import FastNormalDistribution
from utils.kdtree import KDTree
from loguru import logger

np.random.seed(0)


class LocaliseOnTrack:
    def __init__(
        self,
        vehicle_data,
        centre,
        left,
        right,
        cfg,
    ):
        self.vehicle_data = vehicle_data
        distances = np.linalg.norm(centre[1:] - centre[:-1], axis=1)
        self.distance_between_map_points = np.mean(distances)

        self.centre_track = KDTree(centre)
        self.left_track = KDTree(left)
        self.right_track = KDTree(right)

        self.max_number_of_particles = cfg["n_particles"]
        self.number_of_converged_particles = cfg["n_converged_particles"]
        self.sampling_strategy()
        self.best_particle = None

        self.step_count = 0
        self.localised = False

        self.sampling_noise = cfg["sampling_noise"]
        self.control_noise = cfg["control_noise"]

        mean, sigma = (
            cfg["score_distribution"]["mean"],
            cfg["score_distribution"]["sigma"],
        )
        self.distribution = FastNormalDistribution(mean, sigma)
        x = np.linspace(-10, 10, 100)
        self.pdf = self.distribution.pdf
        self.scale = np.max(self.pdf(x))

        self.thresholds = cfg["thresholds"]
        self.convergence_criteria = cfg["convergence_criteria"]

        self.convergence_criteria["maximum_angle"] *= np.pi / 180
        self.thresholds["rotation"] *= np.pi / 180
        self.sampling_noise["yaw"] *= np.pi / 180
        self.control_noise["yaw"] *= np.pi / 180
        self.wheel_base = self.vehicle_data.vehicle_data.wheelbase

    @property
    def n_particles(self) -> int:
        return self.particles["state"].shape[0]

    @property
    def estimated_position(self) -> List[float]:
        weights = self.particles["score"].reshape(-1, 1)
        locations = self.particles["state"][:, :3]
        estimated_location = sum(locations * weights) / sum(weights)
        _, indexes = self._find_closest_points_to_particles(estimated_location[:2])
        centre_idx, left_idx, right_idx = indexes[0], indexes[1], indexes[2]
        return estimated_location, centre_idx, left_idx, right_idx

    @track_runtime
    def step(self, control_command: Tuple[float], dt: float, observation: np.array):
        # maybe_interpolate_track_limit(observation)
        observation = self.downsample_observation(observation)
        self.move_particles(control_command, dt)
        self.update_particles(observation)
        self.step_count += 1

        reset = self.resampling()
        if reset:
            logger.warning("Localisation reset")
            self.update_particles(observation)

        self.is_localised()

    def downsample_observation(self, observation: np.array) -> np.array:
        for i, track in enumerate(observation):
            distances = np.linalg.norm(track[1:] - track[:-1], axis=1)
            downsample = np.mean(distances) / self.distance_between_map_points
            # downsample = max(downsample, 1)
            number_of_points_to_keep = len(track) * (downsample)
            mask = np.zeros(len(track), dtype=np.bool8)
            mask[
                np.linspace(
                    0, len(track) - 1, int(number_of_points_to_keep), dtype=np.uint16
                )
            ] = True
            observation[i] = track[mask]
        return observation

    def move_particles(self, control_input: Tuple[float], dt: float):
        """
        Move the particles based on the control input
            Applies noise to the control input to model uncertainty
            in the movements
            Control input will have velocity
        """
        control_input = (
            -self.vehicle_data.steering_angle(control_input[0]),
            *control_input[1:],
        )
        delta, velocity = self.add_noise_to_control(control_input)
        x_dot = self.calcualte_x_dot(delta, velocity)
        self.advance_particles(dt, x_dot)

    def add_noise_to_control(self, control_input: Tuple[float]) -> Tuple[float]:
        delta = control_input[0] + self.sample_yaw_noise()
        velocity = np.abs(control_input[2] + self.sample_velocity_noise())
        return delta, velocity

    def sample_yaw_noise(self) -> np.array:
        mu, sigma, n = 0, self.control_noise["yaw"], len(self.particles["state"])
        return self.sample_guassian_noise(mu, sigma, n)

    def sample_velocity_noise(self) -> np.array:
        mu, sigma, n = 0, self.control_noise["velocity"], len(self.particles["state"])
        return self.sample_guassian_noise(mu, sigma, n)

    @staticmethod
    def sample_guassian_noise(mu: float, sigma: float, n_samples) -> np.array:
        return np.random.normal(mu, sigma, n_samples)

    def calcualte_x_dot(self, delta: float, velocity: float) -> np.array:
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
        x_dot[:, 2] = velocity * np.tan(delta) / self.wheel_base
        return x_dot

    def advance_particles(self, dt: float, x_dot: np.array):
        self.particles["state"] = self.particles["state"] + dt * x_dot
        self.particles["age"] += 1

    def update_particles(self, observation: List[np.array]):
        """
        Update particle weights based on the observation
        given a map of the track and an observed offset to
        the track
        """
        self.update_particle_closest_points()
        self.update_particle_heading_offsets()
        self.update_particle_error(observation)

    def update_particle_closest_points(self):
        centre_offsets, track_indices = self.find_closest_points_to_particles()
        self.particles["track_indices"] = track_indices
        self.particles["centreline_idx"] = track_indices[:, 0]
        self.particles["minimum_offset"] = centre_offsets

    def find_closest_points_to_particles(self):
        return self._find_closest_points_to_particles(self.particles["state"][:, :2])

    def _find_closest_points_to_particles(self, particle_coordinates):
        _, right_idx = self.right_track.query(particle_coordinates)
        _, left_idx = self.left_track.query(particle_coordinates)
        centre_offsets, centre_idx = self.centre_track.query(particle_coordinates)
        return centre_offsets, np.array([centre_idx, left_idx, right_idx]).T

    def update_particle_heading_offsets(self):
        centreline_idx = self.particles["centreline_idx"]
        centreline_points = self.get_track_points_by_index(centreline_idx)
        next_points = self.get_track_points_by_index(centreline_idx + 1)
        heading_offset = self.calculate_heading_offset(centreline_points, next_points)
        self.particles["heading_offset"] = heading_offset

    def get_track_points_by_index(self, indices: np.array) -> np.array:
        return self.centre_track[np.mod(indices, len(self.centre_track) - 1)]

    def calculate_heading_offset(
        self, centreline_points: np.array, next_points: np.array
    ) -> np.array:
        track_heading = np.arctan2(
            next_points[:, 1] - centreline_points[:, 1],
            next_points[:, 0] - centreline_points[:, 0],
        )
        heading_offset = track_heading - self.particles["state"][:, 2]
        return (heading_offset + np.pi) % (2 * np.pi) - np.pi

    @track_runtime
    def update_particle_error(self, observation: List[np.array]):
        score, error = self.calculate_particle_error(observation)
        self.particles["observation_error"] = error
        self.particles["score"] = score

    def calculate_particle_error(self, observation: List[np.array]):
        observations = self.process_observation(observation)
        track_limits = self.process_track_limits(observation)
        return self.calculate_error(observations, track_limits)

    @track_runtime
    def process_observation(self, observation: List[np.array]) -> np.array:
        # observation[0][:, 0] = np.clip(observation[0][:, 0], -200, 200)
        # observation[0][:, 1] = np.clip(observation[0][:, 1], 0, 50)
        # observation[1][:, 0] = np.clip(observation[1][:, 0], -200, 200)
        # observation[1][:, 1] = np.clip(observation[1][:, 1], 0, 50)
        observation[0] = observation[0][observation[0][:, 1] < 50]
        observation[1] = observation[1][observation[1][:, 1] < 50]
        map_rot = self.calculate_map_rotation()
        observations = np.concatenate(observation)
        n_points = observations.shape[0]
        n_particles = len(self.particles["state"])
        # Repeat observation for each particle the reshape for particles x points x 2 (x,y)
        observations = np.tile(observations, (n_particles, 1))
        observations = observations.reshape(n_particles, -1, 2).transpose(0, 2, 1)
        observations = np.matmul(map_rot, observations)
        observations = observations.transpose(0, 2, 1)
        particle_states = np.tile(
            self.particles["state"][:, :2], (1, n_points)
        ).reshape(n_particles, n_points, 2)
        return np.add(observations, particle_states)

    def calculate_map_rotation(self) -> np.array:
        angle = -self.particles["state"][:, 2] + np.pi / 2
        map_rot = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        # transpose (inverse) since rotation is from observation to particle now
        return map_rot.transpose(2, 1, 0)

    @track_runtime
    def process_track_limits(self, observation: np.array):
        left_track_points = self.get_left_track_limits(observation)
        right_track_points = self.get_right_track_limits(observation)
        return np.concatenate([left_track_points, right_track_points], axis=1)

    def get_left_track_limits(self, observation: np.array) -> np.array:
        closest = self.particles["track_indices"][:, 1]
        return self.get_track_limit(closest, observation[0], self.left_track)

    def get_right_track_limits(self, observation: np.array) -> np.array:
        closest = self.particles["track_indices"][:, 2]
        return self.get_track_limit(closest, observation[1], self.right_track)

    def get_track_limit(
        self, closest: np.array, observation: np.array, track: KDTree
    ) -> np.array:
        n_points = observation.shape[0]
        track_idxs = np.linspace(closest, closest + n_points, n_points, dtype=np.uint16)
        track_idxs = np.mod(track_idxs, len(track)).T
        return track[track_idxs]

    @track_runtime
    def calculate_error(self, observations: np.array, track_limits: np.array):
        average_track_errors = np.mean(
            np.sqrt(np.sum(np.subtract(observations, track_limits) ** 2, axis=2)),
            axis=1,
        )
        score = self.pdf(np.copy(average_track_errors)) / self.scale
        return score, average_track_errors

    def resampling(self):
        """
        This function should take the particles and randomly sample them
        in proportion to their weights, replacing ones lower than a threshold
        """
        self.remove_invalid_particles()
        if self.is_too_few_particles():
            # Lost too many particles, reset
            self.sampling_strategy()
            return True
        self.add_new_particles()
        return False

    def remove_invalid_particles(self):
        valid_particle_mask = self.get_valid_particle_mask()
        for key, val in self.particles.items():
            self.particles[key] = val[valid_particle_mask]

    def get_valid_particle_mask(self) -> np.array:
        sample_mask = (
            # (self.particles["heading_offset"] < self.thresholds["rotation"])
            (self.particles["minimum_offset"] < self.thresholds["offset"])
            & (self.particles["observation_error"] < self.thresholds["track_limit"])
        )
        return sample_mask

    def is_too_few_particles(self) -> bool:
        return self.n_particles < self.thresholds["minimum_particles"]

    def add_new_particles(self):
        n_new_particles = self.get_number_of_new_particles_to_create()
        noise = self.generate_sampling_noise(n_new_particles)
        particle_indices = self.sample_current_particle_indices(n_new_particles)
        new_particle_states = self.particles["state"][particle_indices] + noise
        self.add_new_particle_states(new_particle_states)
        self.add_new_particle_ages(n_new_particles)
        self.add_new_particle_scores(particle_indices)

    def get_number_of_new_particles_to_create(self) -> int:
        n_desired_particles = self.get_desired_number_of_particles()
        return max(0, n_desired_particles - self.n_particles)

    def get_desired_number_of_particles(self) -> int:
        if self.localised:
            return self.number_of_converged_particles
        return self.max_number_of_particles

    def generate_sampling_noise(self, n_samples: int) -> np.array:
        x_noise = self.sample_particle_x_noise(n_samples)
        y_noise = self.sample_particle_y_noise(n_samples)
        yaw_noise = self.sample_particle_yaw_noise(n_samples)
        return np.array([x_noise, y_noise, yaw_noise]).T

    def sample_particle_x_noise(self, n_samples: int) -> np.array:
        mu, sigma = 0, self.sampling_noise["x"]
        return self.sample_guassian_noise(mu, sigma, n_samples)

    def sample_particle_y_noise(self, n_samples: int) -> np.array:
        mu, sigma = 0, self.sampling_noise["y"]
        return self.sample_guassian_noise(mu, sigma, n_samples)

    def sample_particle_yaw_noise(self, n_samples: int) -> np.array:
        mu, sigma = 0, self.sampling_noise["yaw"]
        return self.sample_guassian_noise(mu, sigma, n_samples)

    def sample_current_particle_indices(self, n_samples: int) -> np.array:

        weights = self.particles["score"] / (np.sum(self.particles["score"]))
        return np.random.choice(self.n_particles, size=n_samples, p=weights)

    def add_new_particle_states(self, new_states: np.array):
        updated_states = np.concatenate((self.particles["state"], new_states), axis=0)
        self.particles["state"] = updated_states

    def add_new_particle_ages(self, n_new_particles: int):
        new_ages = np.zeros(n_new_particles)
        updated_ages = np.concatenate((self.particles["age"], new_ages), axis=0)
        self.particles["age"] = updated_ages

    def add_new_particle_scores(self, particle_indices: np.array):
        new_scores = self.particles["score"][particle_indices]
        updated_scores = np.concatenate((self.particles["score"], new_scores), axis=0)
        self.particles["score"] = updated_scores

    def sampling_strategy(self):
        particle_start_point_idx = np.linspace(
            0, len(self.centre_track) - 3, self.max_number_of_particles
        ).astype(np.int16)

        x1 = self.centre_track[particle_start_point_idx, 0]
        y1 = self.centre_track[particle_start_point_idx, 1]
        x2 = self.centre_track[particle_start_point_idx + 1, 0]
        y2 = self.centre_track[particle_start_point_idx + 1, 1]

        yaws = np.arctan2(y2 - y1, x2 - x1)

        samples = np.vstack((x1, y1, yaws)).T

        ages = np.zeros(self.max_number_of_particles)
        scores = np.ones(self.max_number_of_particles)
        scores /= np.sum(scores)

        self.localised = False

        self.particles = {"state": samples, "age": ages, "score": scores}

    def is_localised(self):
        """
        Funciton needs to return a bool of if the filter has converged
        """
        maximum_distance = self.convergence_criteria["maximum_distance"]
        maximum_angle = self.convergence_criteria["maximum_angle"]
        centre = self.estimated_position[0]

        distances_from_centre = np.linalg.norm(
            self.particles["state"][:, :2] - centre[:2], axis=1
        )
        angles_from_estimate = abs(self.particles["state"][:, 2] - centre[2])
        particles_are_close = np.max(distances_from_centre) < maximum_distance
        headings_are_aligned = np.max(angles_from_estimate) < maximum_angle
        if particles_are_close and headings_are_aligned:
            self.localised = True
            return True

        self.localised = False
        return False

    def get_best_weight_info(self):
        best_particle_index = np.argmax(self.particles["score"])
        return (
            self.particles["observation_error"][best_particle_index],
            self.particles["score"][best_particle_index],
        )
