import math
from typing import Dict, List

import numpy as np
import osqp
from loguru import logger
from scipy import sparse

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
        self.current_control = np.zeros((self.nu * self.N))
        self.projected_control = np.zeros((self.nu, self.N))

        # Initialize Optimization Problem
        self.optimizer = osqp.OSQP()

    @property
    def v_max(self) -> float:
        return self.SpeedProfileConstraints["v_max"]

    @property
    def ki_min(self) -> float:
        return self.SpeedProfileConstraints["ki_min"]

    def compute_map_speed_profile(
        self,
        reference_path: List[Dict],
        ay_max: float,
        a_min: float,
    ):
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
        # Inequality Matrix
        D1 = np.zeros((N - 1, N))

        is_bellow_minimum_kappa = np.abs(reference_path.kappas) < self.ki_min
        v_max_dyn = np.sqrt(ay_max / (np.abs(reference_path.kappas) + self.eps))
        logger.info("Breaker")
        v_max_dyn[is_bellow_minimum_kappa] = self.SpeedProfileConstraints["v_max"]
        logger.info("Breaker 2 ")
        logger.debug(f"v_max_dyn: {v_max_dyn}")
        logger.debug(f"v_max: {v_max}")
        logger.debug(f"v_min {v_min}")
        v_mins = np.min([v_max_dyn, v_max], axis=0)
        logger.info(f"v_mins {v_mins}")
        v_maxs = np.max([v_min, v_mins], axis=0)
        logger.info(f"v_maxs {v_maxs}")
        v_max[:] = v_maxs + 2e0

        logger.info("Breaker 3")
        
        lis = reference_path.distances
        logger.info(f"Breaker 3 {np.array([-1 / (2 * lis), 1 / (2 * lis)])}")
        D1[:, :] = np.array([-1 / (2 * lis), 1 / (2 * lis)])

        # Iterate over horizon
        """for i in range(N):
            # Get information about current waypoint
            current_waypoint = reference_path[i]
            # distance between waypoints
            li = current_waypoint["dist_ahead"]
            # curvature of waypoint
            ki = current_waypoint["kappa"]
            # Fill operator matrix
            # dynamics of acceleration
            if i < N - 1:
                D1[i, i : i + 2] = np.array([-1 / (2 * li), 1 / (2 * li)])

            # Compute dynamic constraint on velocity
            if abs(ki) < self.ki_min:
                # Ignore small curvatures caused by noisy homograph
                v_max_dyn = v_max[i]
            else:
                v_max_dyn = np.sqrt(ay_max / (np.abs(ki) + self.eps))

            # + 2.0 creates a feasible regin when v_max == v_min
            v_max[i] = max([v_min[i], min([v_max_dyn, v_max[i]])]) + 2e0
        """
        # Construct inequality matrix
        D1 = sparse.csc_matrix(D1)
        D2 = sparse.eye(N)
        D = sparse.vstack([D1, D2], format="csc")

        # Get upper and lower bound vectors for inequality constraints
        l = np.hstack([a_min, v_min])
        u = np.hstack([a_max, v_max])

        # Set cost matrices
        P = sparse.eye(N, format="csc")
        q = -1 * v_max

        # Solve optimization problem
        problem = osqp.OSQP()
        problem.setup(
            P=P,
            q=q,
            A=D,
            l=l,
            u=u,
            verbose=False,
            max_iter=MAX_SOLVER_ITERATIONS_MAP,
        )
        # speed_profile = problem.solve().x
        dec = problem.solve()
        speed_profile = dec.x

        if dec.info.status == "solved":
            # Assign reference velocity to every waypoint
            reference_path.velocities = speed_profile
            #for i, wp in enumerate(reference_path):
            #    wp["v_ref"] = speed_profile[i]
            self.speed_profile = speed_profile
            return reference_path

        else:
            message = f"Infeasible problem! reference path:\n"
            failed_reference_path = np.hstack([reference_path.xs, reference_path.ys])
            logger.warning(message + f"{failed_reference_path}")
            return reference_path

    def compute_speed_profile(self, reference_path: List[Dict], end_vel=None):
        """
        Compute a speed profile for the path. Assign a reference velocity
        to each waypoint based on its curvature.
        :param Constraints: constraints on acceleration and velocity
        curvature of the path
        """

        # Set optimization horizon
        N = len(reference_path)
        # Constraints
        a_max = np.ones(N - 1) * self.SpeedProfileConstraints["a_max"]
        v_min = np.ones(N) * self.SpeedProfileConstraints["v_min"]
        v_max = np.ones(N) * self.SpeedProfileConstraints["v_max"]
        ay_max = self.SpeedProfileConstraints["ay_max"]
        a_min = np.ones(N - 1) * self.SpeedProfileConstraints["a_min"]
        # Inequality Matrix
        D1 = np.zeros((N - 1, N))

        # Iterate over horizon
        for i in range(N):
            # Get information about current waypoint
            current_waypoint = reference_path[i]
            # distance between waypoints
            li = current_waypoint["dist_ahead"]
            # curvature of waypoint
            ki = current_waypoint["kappa"]
            # Fill operator matrix
            # dynamics of acceleration
            if i < N - 1:
                D1[i, i : i + 2] = np.array([-1 / (2 * li), 1 / (2 * li)])

            # Compute dynamic constraint on velocity
            if abs(ki) < self.ki_min:
                # Ignore small curvatures caused by noisy homograph
                v_max_dyn = v_max[i]
            else:
                v_max_dyn = np.sqrt(ay_max / (np.abs(ki) + self.eps))

            # + 2.0 creates a feasible regin when v_max == v_min
            v_max[i] = max([v_min[i], min([v_max_dyn, v_max[i]])]) + 2e0

        if end_vel:
            v_max[-1] = min(end_vel, v_max[-1])

        # Construct inequality matrix
        D1 = sparse.csc_matrix(D1)
        D2 = sparse.eye(N)
        D = sparse.vstack([D1, D2], format="csc")

        # Get upper and lower bound vectors for inequality constraints
        l = np.hstack([a_min, v_min])
        u = np.hstack([a_max, v_max])

        # Set cost matrices
        P = sparse.eye(N, format="csc")
        q = -1 * v_max

        # Solve optimization problem
        problem = osqp.OSQP()
        problem.setup(
            P=P,
            q=q,
            A=D,
            l=l,
            u=u,
            verbose=False,
            max_iter=MAX_SOLVER_ITERATIONS,
        )
        # speed_profile = problem.solve().x
        dec = problem.solve()
        speed_profile = dec.x

        if dec.info.status == "solved":

            # Assign reference velocity to every waypoint
            for i, wp in enumerate(reference_path):
                wp["v_ref"] = speed_profile[i]

            self.speed_profile = speed_profile
            return reference_path

        else:
            message = f"Infeasible problem! reference path:\n"
            failed_reference_path = np.array(
                [[val["x"], val["y"]] for i, val in enumerate(reference_path)]
            )
            logger.warning(message + f"{failed_reference_path}")
            return reference_path

    def construct_waypoints(self, waypoint_coordinates: np.array) -> List[Dict]:
        """
        Reformulate conventional waypoints (x, y) coordinates into waypoint
        objects containing (x, y, psi, kappa, ub, lb)
        :param waypoint_coordinates: list of (x, y) coordinates of waypoints in
        global coordinates
        :return: list of waypoint objects for entire reference path
        """

        # List containing waypoint objects
        n_points = len(waypoint_coordinates) - 2
        waypoints = ReferencePath(n_points)

        previous_wps = waypoint_coordinates[0:-2, :-1]
        current_wps = waypoint_coordinates[1:-1, :-1]
        next_wps = waypoint_coordinates[2:, :-1]

        diffs_ahead = current_wps - next_wps
        diffs_behind = current_wps - previous_wps

        waypoints.xs = waypoint_coordinates[1:-1, 0]
        waypoints.ys = waypoint_coordinates[1:-1, 1]
        waypoints.widths = waypoint_coordinates[1:-1, 2]
        waypoints.psis = np.arctan2(diffs_ahead[:, 1], diffs_ahead[:, 0])
        waypoints.distances = np.sqrt(diffs_ahead[:, 0] ** 2 + diffs_ahead[:, 1] ** 2)

        angles_behind = np.arctan2(diffs_behind[:, 1], diffs_behind[:, 0])
        angle_diffs = (
            np.mod(waypoints.psis - angles_behind + math.pi, 2 * math.pi) - math.pi
        )
        waypoints.kappas = angle_diffs / (waypoints.distances + self.eps)

        #    if wp_id == 0:
        #        kappa = 0
        #    elif wp_id == 1:
        #        waypoints[0]["kappa"] = kappa
        # logger.debug(waypoints.kappas)
        return waypoints

    def update_prediction(self, spatial_state_prediction, reference_path):
        """
        Transform the predicted states to predicted x and y coordinates.
        Mainly for visualization purposes.
        :param spatial_state_prediction: list of predicted state variables
        :return: lists of predicted x and y coordinates
        """

        # Containers for x and y coordinates of predicted states
        predicted_locations = np.zeros((self.N, 2))

        # Iterate over prediction horizon
        for n in range(self.N):
            # Get associated waypoint
            associated_waypoint = reference_path[n]
            # Transform predicted spatial state to temporal state
            predicted_temporal_state = self.model.s2t(
                associated_waypoint, spatial_state_prediction[n, :]
            )

            # Save predicted coordinates in world coordinate frame
            predicted_locations[n, :] = predicted_temporal_state[:-1]

        return predicted_locations

    def _init_problem(self, spatial_state, reference_path):
        """
        Initialize optimization problem for current time step.
        """
        self.N = len(reference_path)

        # Constraints
        umin = self.input_constraints["umin"]
        umax = self.input_constraints["umax"]
        xmin = self.state_constraints["xmin"]
        xmax = self.state_constraints["xmax"]

        # LTV System Matrices
        A = np.zeros((self.nx * (self.N + 1), self.nx * (self.N + 1)))
        B = np.zeros((self.nx * (self.N + 1), self.nu * (self.N)))
        # Reference vector for state and input variables
        ur = np.zeros(self.nu * self.N)
        xr = np.zeros(self.nx * (self.N + 1))
        # Offset for equality constraint (due to B * (u - ur))
        uq = np.zeros(self.N * self.nx)
        # Dynamic state constraints
        xmin_dyn = np.kron(np.ones(self.N + 1), xmin)
        xmax_dyn = np.kron(np.ones(self.N + 1), xmax)
        # Dynamic input constraints
        umax_dyn = np.kron(np.ones(self.N), umax)
        umin_dyn = np.kron(np.ones(self.N), umin)
        # umax_dyn[:2] = 0
        # Get curvature predictions from previous control signals
        kappa_pred = (
            np.tan(np.array(self.current_control[3::] + self.current_control[-1:]))
            / self.model.length
        )

        # Iterate over horizon
        for n in range(self.N):
            # Get information about current waypoint
            current_waypoint = reference_path[n]
            delta_s = current_waypoint["dist_ahead"]
            kappa_ref = current_waypoint["kappa"]
            v_ref = current_waypoint["v_ref"]

            # Compute LTV matrices
            f, A_lin, B_lin = self.model.linearize(v_ref, kappa_ref, delta_s)
            A[
                (n + 1) * self.nx : (n + 2) * self.nx, n * self.nx : (n + 1) * self.nx
            ] = A_lin
            B[
                (n + 1) * self.nx : (n + 2) * self.nx, n * self.nu : (n + 1) * self.nu
            ] = B_lin

            # Set reference for input signal
            ur[n * self.nu : (n + 1) * self.nu] = np.array([v_ref, kappa_ref])
            # Compute equality constraint offset (B*ur)
            uq[n * self.nx : (n + 1) * self.nx] = (
                B_lin.dot(np.array([v_ref, kappa_ref])) - f
            )

            # Constrain maximum speed based on predicted car curvature
            vmax_dyn = np.sqrt(self.ay_max / (np.abs(kappa_pred[n]) + 1e-12))
            # if vmax_dyn < umax_dyn[self.nu * n]:
            #     umax_dyn[self.nu * n] = vmax_dyn

            umax_dyn[self.nu * n] = min([vmax_dyn, umax_dyn[self.nu * n], v_ref]) + 1e-1
            umin_dyn[self.nu * n] = min([vmax_dyn, umin_dyn[self.nu * n], v_ref]) - 1e-1

        ub = (
            np.array([reference_path[i]["width"] / 2 for i in range(self.N)])
            - self.model.safety_margin
        )
        lb = (
            np.array([-reference_path[i]["width"] / 2 for i in range(self.N)])
            + self.model.safety_margin
        )
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
        lineq = np.hstack([xmin_dyn, umin_dyn])
        uineq = np.hstack([xmax_dyn, umax_dyn])
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
                -np.tile(np.diag(self.Q.A), self.N) * xr[: -self.nx],
                -self.QN.dot(xr[-self.nx :]),
                -np.tile(np.diag(self.R.A), self.N) * ur,
            ]
        )

        # Initialize optimizer
        self.optimizer = osqp.OSQP()
        self.optimizer.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)

    def get_control(self, reference_path, offset=0):
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
        spatial_state = self.model.t2s(
            reference_state=state, reference_waypoint=self.reference_path[0]
        )

        # Initialize optimization problem
        self._init_problem(spatial_state, self.reference_path)

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

            # self.current_control = control_signals

            # Get predicted spatial states
            x = np.reshape(dec.x[: (self.N + 1) * nx], (self.N + 1, nx))

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
            # failed_reference_path = np.array(
            #    [[val["x"], val["y"]] for i, val in enumerate(self.reference_path)]
            # )
            logger.warning(message)  # + f"{failed_reference_path}")
            self.infeasibility_counter += 1
