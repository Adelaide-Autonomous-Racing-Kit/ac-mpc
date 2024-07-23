import math
from typing import Dict, List

import numpy as np
import osqp
from loguru import logger
from scipy import sparse

import sys

np.set_printoptions(threshold=sys.maxsize)

from control.paths import ReferencePath

MAX_SOLVER_ITERATIONS_MAP = 40000
MAX_SOLVER_ITERATIONS = 4000


class SpatialMPC:
    def __init__(
        self,
        model,
        delta_max,
        N,
        Q,
        R,
        QN,
        StateConstraints,
        InputConstraints,
        SpeedProfileConstraints,
    ):
        """
        Constructor for the Model Predictive Controller.
        :param model: bicycle model object to be controlled
        :param N: time horizon | int
        :param Q: state cost matrix
        :param R: input cost matrix
        :param QN: final state cost matrix
        :param StateConstraints: dictionary of state constraints
        :param InputConstraints: dictionary of input constraints
        :param ay_max: maximum allowed lateral acceleration in curves
        """
        # Parameters
        self.MPC_horizon = N
        self.N = N  # horizon
        self.Q = Q  # weight matrix state vector
        self.R = R  # weight matrix input vector
        self.QN = QN  # weight matrix terminal
        self.cum_time = np.zeros((1))

        # Model
        self.model = model

        # Dimensions
        self.nx = self.model.n_states
        self.nu = 2

        # Precision
        self.eps = 1e-12

        # Constraints
        self.state_constraints = StateConstraints
        self.input_constraints = InputConstraints

        self.SpeedProfileConstraints = SpeedProfileConstraints

        # Maximum lateral acceleration
        self.ay_max = self.SpeedProfileConstraints["ay_max"]

        # Maximum steering angle
        self.delta_max = delta_max

        # Current control and prediction
        self.current_prediction = None

        # Counter for old control signals in case of infeasible problem
        self.infeasibility_counter = 0

        # Current control signals
        self.projected_control = np.zeros((self.nu, self.N))

        # Initialize Optimization Problem
        self.optimizer = None

    @property
    def v_max(self) -> float:
        return self.SpeedProfileConstraints["v_max"]

    @property
    def ki_min(self) -> float:
        return self.SpeedProfileConstraints["ki_min"]

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
        # Set optimization horizon
        N = len(reference_path)
        # Constraints
        a_max = np.ones(N - 1) * self.SpeedProfileConstraints["a_max"]
        v_min = np.ones(N) * self.SpeedProfileConstraints["v_min"]
        v_max = np.ones(N) * self.SpeedProfileConstraints["v_max"]
        a_min = np.ones(N - 1) * a_min
        # Maximum velocities
        is_bellow_minimum_kappa = np.abs(reference_path.kappas) < self.ki_min
        v_max_dyn = np.sqrt(ay_max / (np.abs(reference_path.kappas) + self.eps))
        v_max_dyn[is_bellow_minimum_kappa] = self.SpeedProfileConstraints["v_max"]
        v_mins = np.min([v_max_dyn, v_max], axis=0)
        v_maxs = np.max([v_min, v_mins], axis=0)
        v_max = v_maxs + 2e0
        # Inequality Matrix
        lis = reference_path.distances
        D1_diagonal = np.array([-1 / (2 * lis[:-1]), 1 / (2 * lis[:-1])])
        D1 = sparse.diags(D1_diagonal, offsets=[0, 1], shape=[N - 1, N])

        # Construct inequality matrix
        D1 = sparse.csc_matrix(D1)
        D2 = sparse.eye(N)
        D = sparse.vstack([D1, D2], format="csc")

        # Get upper and lower bound vectors for inequality constraints
        lower_bound = np.hstack([a_min, v_min])
        upper_bound = np.hstack([a_max, v_max])

        # Set cost matrices
        P = sparse.eye(N, format="csc")
        q = -1 * v_max

        # Solve optimization problem
        problem = osqp.OSQP()
        problem.setup(
            P=P,
            q=q,
            A=D,
            l=lower_bound,
            u=upper_bound,
            verbose=False,
            max_iter=MAX_SOLVER_ITERATIONS_MAP,
        )
        dec = problem.solve()
        speed_profile = dec.x

        if dec.info.status == "solved":
            # Assign reference velocity to every waypoint
            reference_path.velocities = speed_profile
            self.speed_profile = speed_profile
            return reference_path

        else:
            message = "Infeasible problem! reference path:\n"
            failed_reference_path = np.hstack([reference_path.xs, reference_path.ys])
            logger.warning(message + f"{failed_reference_path}")
            return reference_path

    def compute_speed_profile(
        self,
        reference_path: ReferencePath,
        end_vel=None,
    ) -> ReferencePath:
        """
        Compute a speed profile for the path. Assign a reference velocity
        to each waypoint based on its curvature.
        :param Constraints: constraints on acceleration and velocity
        curvature of the path
        """
        # TODO: Remove duplication between here and compute_map_speed_profile
        # Set optimization horizon
        N = len(reference_path)
        # Constraints
        a_max = np.ones(N - 1) * self.SpeedProfileConstraints["a_max"]
        v_min = np.ones(N) * self.SpeedProfileConstraints["v_min"]
        v_max = np.ones(N) * self.SpeedProfileConstraints["v_max"]
        a_min = np.ones(N - 1) * self.SpeedProfileConstraints["a_min"]
        ay_max = self.SpeedProfileConstraints["ay_max"]
        # Maximum velocities
        # is_bellow_minimum_kappa = np.abs(reference_path.kappas) < self.ki_min
        # v_max_dyn = np.sqrt(ay_max / (np.abs(reference_path.kappas) + self.eps))
        # v_max_dyn[is_bellow_minimum_kappa] = self.SpeedProfileConstraints["v_max"]
        # v_mins = np.min([v_max_dyn, v_max], axis=0)
        # v_maxs = np.max([v_min, v_mins], axis=0)
        # v_max = v_maxs + 2e0
        # if end_vel is not None:
        #    v_max[-1] = end_vel
        # Inequality Matrix
        lis = reference_path.distances
        D1_diagonal = np.array([-1 / (2 * lis[:-1]), 1 / (2 * lis[:-1])])
        D1 = sparse.diags(D1_diagonal, offsets=[0, 1], shape=[N - 1, N])

        # Construct inequality matrix
        D2 = sparse.eye(N)
        D = sparse.vstack([D1, D2], format="csc")

        # Get upper and lower bound vectors for inequality constraints
        lower_bound = np.hstack([a_min, v_min])
        upper_bound = np.hstack([a_max, v_max])

        # Set cost matrices
        P = sparse.eye(N, format="csc")
        q = -1 * v_max

        # Solve optimization problem
        problem = osqp.OSQP()
        problem.setup(
            P=P,
            q=q,
            A=D,
            l=lower_bound,
            u=upper_bound,
            verbose=False,
            max_iter=MAX_SOLVER_ITERATIONS,
        )
        dec = problem.solve()
        speed_profile = dec.x

        if dec.info.status == "solved":
            # Assign reference velocity to every waypoint
            reference_path.velocities = speed_profile
            self.speed_profile = speed_profile
            return reference_path
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
        kappas = angle_diffs / (waypoints.distances + self.eps)
        kappas[0] = kappas[1]
        waypoints.kappas = kappas
        return waypoints

    def update_prediction(
        self, spatial_state_prediction: np.array, reference_path: ReferencePath
    ) -> np.array:
        """
        Transform the predicted states to predicted x and y coordinates.
        Mainly for visualization purposes.
        :param spatial_state_prediction: list of predicted state variables
        :return: lists of predicted x and y coordinates
        """
        predicted_locations = self.model.s2t(reference_path, spatial_state_prediction)
        return predicted_locations[:-1].T

    def init_problem(self, spatial_state, reference_path):
        """
        Initialize optimization problem for current time step.
        """
        self.N = len(reference_path)

        # Constraints
        umin = self.input_constraints["umin"]
        umax = self.input_constraints["umax"]
        xmin = self.state_constraints["xmin"]
        xmax = self.state_constraints["xmax"]

        xr = np.zeros(self.nx * (self.N + 1))
        # Dynamic state constraints
        xmin_dyn = np.kron(np.ones(self.N + 1), xmin)
        xmax_dyn = np.kron(np.ones(self.N + 1), xmax)
        # Dynamic input constraints
        umax = np.kron(np.ones(self.N), umax)
        umin = np.kron(np.ones(self.N), umin)
        umax[::2] += 0.1
        umin[::2] -= 0.1

        # Compute LTV matrices
        v_refs = reference_path.velocities
        refs = np.array([reference_path.velocities, reference_path.kappas])
        f, A, B = self.model.linearise(reference_path)
        # Reference vector for state and input variables
        ur = np.ravel(refs, order="F")
        # Offset for equality constraint (due to B * (u - ur))
        uq = np.ravel(np.einsum("BNi,iB ->BN", B, refs) - f)
        # Format matrices
        A = sparse.block_diag(A, format="csc")
        A = sparse.block_array([[np.zeros((3, A.shape[1]))], [A]], format="csc")
        A = sparse.block_array([[A, np.zeros((A.shape[0], 3))]], format="csc")
        B = sparse.block_diag(B, format="csc")
        B = sparse.block_array([[np.zeros((3, B.shape[1]))], [B]], format="csc")

        ub = (reference_path.widths / 2) - self.model.safety_margin
        lb = (-reference_path.widths / 2) + self.model.safety_margin

        xmin_dyn[0] = spatial_state[0]
        xmax_dyn[0] = spatial_state[0]
        xmin_dyn[self.nx :: self.nx] = lb
        xmax_dyn[self.nx :: self.nx] = ub
        # Set reference for state as center-line of drivable area
        xr[self.nx :: self.nx] = (lb + ub) / 2

        # Get equality matrix
        Ax = sparse.kron(
            sparse.eye(self.N + 1), -sparse.eye(self.nx)
        ) + sparse.csc_matrix(A)
        Bu = sparse.csc_matrix(B)
        Aeq = sparse.hstack([Ax, Bu])
        # Get inequality matrix
        Aineq = sparse.eye((self.N + 1) * self.nx + self.N * self.nu)
        # Combine constraint matrices
        A = sparse.vstack([Aeq, Aineq], format="csc")

        # Get upper and lower bound vectors for equality constraints
        lineq = np.hstack([xmin_dyn, umin])
        uineq = np.hstack([xmax_dyn, umax])
        # Get upper and lower bound vectors for inequality constraints
        x0 = np.array(spatial_state)
        leq = np.hstack([-x0, uq])
        ueq = leq
        # Combine upper and lower bound vectors
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        # Set cost matrices
        P = sparse.block_diag(
            [
                sparse.kron(sparse.eye(self.N), self.Q),
                self.QN,
                sparse.kron(sparse.eye(self.N), self.R),
            ],
            format="csc",
        )
        q = np.hstack(
            [
                -np.tile(np.diag(self.Q.toarray()), self.N) * xr[: -self.nx],
                -self.QN.dot(xr[-self.nx :]),
                -np.tile(np.diag(self.R.toarray()), self.N) * ur,
            ]
        )

        # Initialize optimizer
        self.optimizer = osqp.OSQP()
        self.optimizer.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)

    def get_control(self, reference_path: np.array, offset: float = 0.0):
        """
        Get control signal given the current position of the car.
        Solves a finite time optimization problem based on the linearized car model.
        """
        self.reference_path = self.construct_waypoints(reference_path)
        self.reference_path = self.compute_speed_profile(
            self.reference_path, end_vel=self.SpeedProfileConstraints["end_velocity"]
        )

        # Number of state variables
        nx = self.model.n_states
        nu = 2

        # x, y psi (y axis is forward)
        state = np.array([offset, 0, np.pi / 2])
        # Update spatial state
        spatial_state = self.model.t2s(self.reference_path.get_state(0), state)

        # Initialize optimization problem
        self.init_problem(spatial_state, self.reference_path)

        # Solve optimization problem
        dec = self.optimizer.solve()

        if dec.info.status == "solved":
            # Get control signals
            control_signals = np.array(dec.x[-self.N * nu :])
            control_signals[1::2] = np.arctan(control_signals[1::2] * self.model.length)

            # Update control signals
            all_velocities = control_signals[0::2]
            all_delta = control_signals[1::2]
            self.projected_control = np.array([all_velocities, all_delta])

            # Get predicted spatial states
            x = np.reshape(dec.x[: (self.N) * nx], (self.N, nx))

            # Update predicted temporal states
            self.current_prediction = self.update_prediction(x, self.reference_path)

            self.cum_time = x[:, 2]
            self.times = np.diff(x[:, 2])

            self.accelerations = np.diff(x[:, 0]) / self.times
            self.steer_rates = np.diff(x[:, 1]) / self.times
            # if problem solved, reset infeasibility counter
            self.infeasibility_counter = 0

        else:
            n_times_failed = self.infeasibility_counter
            message = f"Infeasible problem! Failed {n_times_failed} time(s)."
            logger.warning(message)
            self.infeasibility_counter += 1
