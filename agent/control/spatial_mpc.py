import math

import numpy as np
import osqp
from loguru import logger
from scipy import sparse


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

    def compute_speed_profile(self, reference_path, end_vel=None):
        """
        Compute a speed profile for the path. Assign a reference velocity
        to each waypoint based on its curvature.
        :param Constraints: constraints on acceleration and velocity
        curvature of the path
        """

        # Set optimization horizon
        N = len(reference_path)

        # Constraints
        a_min = np.ones(N - 1) * self.SpeedProfileConstraints["a_min"]
        a_max = np.ones(N - 1) * self.SpeedProfileConstraints["a_max"]
        v_min = np.ones(N) * self.SpeedProfileConstraints["v_min"]
        v_max = np.ones(N) * self.SpeedProfileConstraints["v_max"]

        # Maximum lateral acceleration
        ay_max = self.SpeedProfileConstraints["ay_max"]

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
            v_max_dyn = np.sqrt(ay_max / (np.abs(ki) + self.eps))
            # if v_max_dyn < v_max[i]:
            #     v_max[i] = v_max_dyn

            v_max[i] = min([v_max_dyn, v_max[i]]) +2e0
            v_min[i] = min([v_max_dyn, v_min[i]]) -2e0

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
        problem.setup(P=P, q=q, A=D, l=l, u=u, verbose=False)
        speed_profile = problem.solve().x

        # Assign reference velocity to every waypoint
        for i, wp in enumerate(reference_path):
            wp["v_ref"] = speed_profile[i]

        self.speed_profile = speed_profile
        return reference_path

    def construct_waypoints(self, waypoint_coordinates):
        """
        Reformulate conventional waypoints (x, y) coordinates into waypoint
        objects containing (x, y, psi, kappa, ub, lb)
        :param waypoint_coordinates: list of (x, y) coordinates of waypoints in
        global coordinates
        :return: list of waypoint objects for entire reference path
        """

        # List containing waypoint objects
        waypoints = []

        # Iterate over all waypoints
        for wp_id in range(len(waypoint_coordinates) - 1):
            # Get start and goal waypoints
            current_wp = np.array(waypoint_coordinates[wp_id])[:-1]
            next_wp = np.array(waypoint_coordinates[wp_id + 1])[:-1]
            width = np.array(waypoint_coordinates[wp_id + 1])[-1]

            # Difference vector
            dif_ahead = next_wp - current_wp

            # Angle ahead
            psi = np.arctan2(dif_ahead[1], dif_ahead[0])

            # Distance to next waypoint
            dist_ahead = np.sqrt(dif_ahead[0] ** 2 + dif_ahead[1] ** 2)

            # Get x and y coordinates of current waypoint
            x, y = current_wp[0], current_wp[1]

            # Compute local curvature at waypoint
            # first waypoint

            prev_wp = np.array(waypoint_coordinates[wp_id - 1][:-1])
            dif_behind = current_wp - prev_wp
            angle_behind = np.arctan2(dif_behind[1], dif_behind[0])
            angle_dif = np.mod(psi - angle_behind + math.pi, 2 * math.pi) - math.pi
            kappa = angle_dif / (dist_ahead + self.eps)

            if wp_id == 0:
                kappa = 0
            elif wp_id == 1:
                waypoints[0]["kappa"] = kappa

            waypoints.append(
                {
                    "x": x,
                    "y": y,
                    "psi": psi,
                    "kappa": kappa,
                    "dist_ahead": dist_ahead,
                    "width": width,
                }
            )

        return waypoints

    def update_prediction(self, spatial_state_prediction, reference_path):
        """
        Transform the predicted states to predicted x and y coordinates.
        Mainly for visualization purposes.
        :param spatial_state_prediction: list of predicted state variables
        :return: lists of predicted x and y coordinates
        """

        # Containers for x and y coordinates of predicted states
        x_pred, y_pred = [], []

        # Iterate over prediction horizon
        for n in range(self.N):
            # Get associated waypoint
            associated_waypoint = reference_path[n]
            # Transform predicted spatial state to temporal state
            predicted_temporal_state = self.model.s2t(
                associated_waypoint, spatial_state_prediction[n, :]
            )

            # Save predicted coordinates in world coordinate frame
            x_pred.append(predicted_temporal_state[0])
            y_pred.append(predicted_temporal_state[1])

        return x_pred, y_pred

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
        Get control signal given the current position of the car. Solves a
        finite time optimization problem based on the linearized car model.
        """
        self.reference_path = self.construct_waypoints(reference_path)
        self.reference_path = self.compute_speed_profile(
            self.reference_path, end_vel=10.0
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
            v = control_signals[2]
            delta = control_signals[3]

            # Update control signals
            all_velocities = control_signals[0::2]
            all_delta = control_signals[1::2]
            self.projected_control = np.array([all_velocities, all_delta])

            # self.current_control = control_signals

            # Get predicted spatial states
            x = np.reshape(dec.x[: (self.N + 1) * nx], (self.N + 1, nx))

            # Update predicted temporal states
            self.current_prediction = self.update_prediction(x, self.reference_path)

            self.times = np.diff(x[:, 2])
            self.cum_time = np.cumsum(self.times)

            self.accelerations = np.diff(x[:, 0]) / self.times
            self.steer_rates = np.diff(x[:, 1]) / self.times

            # Get current control signal
            u = np.array([v, delta])

            # if problem solved, reset infeasibility counter
            self.infeasibility_counter = 0

        else:
            message = "Infeasible problem. Previous control signal used!\n"
            failed_reference_path = np.array(
                [[val["x"], val["y"]] for i, val in enumerate(self.reference_path)]
            )
            logger.warning(message + f"{failed_reference_path}")
            id = nu * (self.infeasibility_counter + 1)
            u = np.array(self.current_control[id : id + 2])

            # increase infeasibility counter
            self.infeasibility_counter += 1

        if self.infeasibility_counter == (self.N - 1):
            logger.error("No control signal computed!")
            exit(1)

        return u
