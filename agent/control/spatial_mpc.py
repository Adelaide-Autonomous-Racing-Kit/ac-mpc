from __future__ import annotations

import copy
import math
from typing import Dict

from control.paths import ReferencePath
from control.solvers import (
    ControlSolver,
    LocalisedSpeedProfileSolver,
    SpeedProfileSolver,
)
from loguru import logger
import numpy as np

MAX_SOLVER_ITERATIONS_MAP = 40000
MAX_SOLVER_ITERATIONS = 4000


class SpatialMPC:
    def __init__(self, config: Dict, model: SpatialBicycleModel):
        """
        Constructor for the Model Predictive Controller.
        :param config: dictionary of configuration options
        :param model: bicycle model object to be controlled
        """
        self.MPC_horizon = config["horizon"]
        self.cum_time = np.zeros((1))
        self.model = model
        # Dimensions
        self.nx = 3
        self.nu = 2
        # Precision
        self._eps = 1e-12
        # Constraints
        self.speed_profile_constraints = config["speed_profile_constraints"]
        # Maximum lateral acceleration
        self.ay_max = self.speed_profile_constraints["ay_max"]
        # Maximum steering angle
        self.delta_max = model.delta_max
        # Current control and prediction
        self.current_prediction = None
        # Counter for old control signals in case of infeasible problem
        self.infeasibility_counter = 0
        # Current control signals
        self.projected_control = np.zeros((self.nu, self.MPC_horizon))
        # Control Solver
        config = copy.deepcopy(config)
        config["max_iterations"] = MAX_SOLVER_ITERATIONS
        self._control_solver = ControlSolver(config, model)
        # Speed Profile Solvers
        config = {
            "control_horizon": self.MPC_horizon - 1,
            "max_iterations": MAX_SOLVER_ITERATIONS,
            "constraints": self.speed_profile_constraints,
        }
        self._speed_profile_solver = SpeedProfileSolver(config)
        self._localised_speed_profile_solver = LocalisedSpeedProfileSolver(config)

    def compute_map_speed_profile(
        self,
        reference_path: ReferencePath,
        ay_max: float,
        a_min: float,
    ) -> ReferencePath:
        """
        Compute a speed profile for the path. Assign a reference velocity
        to each waypoint based on its curvature.
        """
        solver = self._setup_map_speed_profile_solver(reference_path, ay_max, a_min)
        return self._compute_speed_profile(solver, reference_path)

    def _setup_map_speed_profile_solver(
        self,
        reference_path: ReferencePath,
        ay_max: float,
        a_min: float,
    ) -> SpeedProfileSolver:
        constraints = copy.deepcopy(self.speed_profile_constraints)
        constraints["a_min"] = a_min
        constraints["ay_max"] = ay_max
        config = {
            "control_horizon": len(reference_path),
            "max_iterations": MAX_SOLVER_ITERATIONS_MAP,
            "constraints": constraints,
        }
        return SpeedProfileSolver(config)

    def compute_speed_profile(
        self,
        reference_path: ReferencePath,
        is_localised: bool = False,
        end_vel=None,
    ) -> ReferencePath:
        """
        Compute a speed profile for the path. Assign a reference velocity
        to each waypoint based on its curvature.
        :param Constraints: constraints on acceleration and velocity
        curvature of the path
        """
        if is_localised:
            solver = self._localised_speed_profile_solver
        else:
            solver = self._speed_profile_solver
        return self._compute_speed_profile(solver, reference_path, end_vel)

    def _compute_speed_profile(
        self,
        solver: SpeedProfileSolver,
        reference_path: ReferencePath,
        end_vel=None,
    ) -> ReferencePath:
        dec = solver.solve(reference_path, end_vel)
        speed_profile = dec.x
        if dec.info.status == "solved":
            # Assign reference velocity to every waypoint
            reference_path.velocities = speed_profile
            self.speed_profile = speed_profile
        else:
            message = "Infeasible problem! reference path:\n"
            failed_reference_path = np.hstack([reference_path.xs, reference_path.ys])
            logger.warning(message + f"{failed_reference_path}")
        return reference_path

    def construct_waypoints(self, waypoint_coordinates: np.array) -> ReferencePath:
        """
        Reformulate conventional waypoints (x, y) coordinates into waypoint
        objects containing (x, y, psi, kappa, ub, lb)
        :param waypoint_coordinates: list of (x, y) coordinates of waypoints in
        global coordinates
        :return: list of waypoint objects for entire reference path
        """
        n_points = len(waypoint_coordinates) - 1
        waypoints = ReferencePath(n_points)
        previous_wps = np.vstack(
            [waypoint_coordinates[-1, :-1], waypoint_coordinates[:-2, :-1]]
        )
        current_wps = waypoint_coordinates[:-1, :-1]
        next_wps = waypoint_coordinates[1:, :-1]
        diffs_ahead = next_wps - current_wps
        diffs_behind = current_wps - previous_wps
        waypoints.xs = waypoint_coordinates[:-1, 0]
        waypoints.ys = waypoint_coordinates[:-1, 1]
        waypoints.widths = waypoint_coordinates[1:, 2]
        waypoints.psis = np.arctan2(diffs_ahead[:, 1], diffs_ahead[:, 0])
        waypoints.distances = np.linalg.norm(diffs_ahead, axis=1)
        # Kappa calculation
        angles_behind = np.arctan2(diffs_behind[:, 1], diffs_behind[:, 0])
        angle_diffs = waypoints.psis - angles_behind + math.pi
        angle_diffs = np.mod(angle_diffs, 2 * math.pi) - math.pi
        kappas = angle_diffs / (waypoints.distances + self._eps) + self._eps
        kappas[0] = kappas[1]
        waypoints.kappas = kappas
        return waypoints

    def update_prediction(
        self,
        spatial_state_prediction: np.array,
        reference_path: ReferencePath,
    ) -> np.array:
        """
        Transform the predicted states to predicted x and y coordinates.
        Mainly for visualization purposes.
        :param spatial_state_prediction: list of predicted state variables
        :return: lists of predicted x and y coordinates
        """
        predicted_locations = self.model.s2t(reference_path, spatial_state_prediction)
        return predicted_locations[:-1].T

    def get_control(
        self,
        reference_path: np.array,
        is_localised: bool = False,
        offset: float = 0.0,
    ):
        """
        Get control signal given the current position of the car.
        Solves a finite time optimization problem based on the linearized car model.
        """
        reference_path = self.construct_waypoints(reference_path)
        reference_path = self.compute_speed_profile(
            reference_path,
            is_localised,
            end_vel=self.speed_profile_constraints["end_velocity"],
        )
        # x, y psi (y axis is forward)
        state = np.array([offset, 0, np.pi / 2])
        # Update spatial state
        spatial_state = self.model.t2s(reference_path.get_state(0), state)
        # Initialize optimization problem
        dec = self._control_solver.solve(spatial_state, reference_path)

        if dec.info.status == "solved":
            # Get control signals
            control_signals = np.array(dec.x[-(self.MPC_horizon - 1) * self.nu :])
            control_signals[1::2] = np.arctan(control_signals[1::2] * self.model.length)
            # Update control signals
            all_velocities = control_signals[0::2]
            all_delta = control_signals[1::2]
            self.projected_control = np.array([all_velocities, all_delta])
            # Get predicted spatial states
            states = dec.x[: (self.MPC_horizon - 1) * self.nx]
            shape = (self.MPC_horizon - 1, self.nx)
            x = np.reshape(states, shape)
            # Update predicted temporal states
            self.current_prediction = self.update_prediction(x, reference_path)
            self.reference_path = reference_path
            self.cum_time = x[:, 2]
            self.times = np.diff(x[:, 2])
            self.accelerations = np.diff(x[:, 0]) / self.times
            self.steer_rates = np.diff(x[:, 1]) / self.times
            self.infeasibility_counter = 0
        else:
            n_times_failed = self.infeasibility_counter
            message = f"Infeasible problem! Failed {n_times_failed} time(s)."
            logger.warning(message)
            self.infeasibility_counter += 1
