from __future__ import annotations
from typing import Dict, Tuple
import math

import numpy as np


class SpatialBicycleModel:
    def __init__(self, vehicle_data: SteeringGeometry, velocity_limits: Dict):
        self.length = vehicle_data.vehicle_data.wheelbase
        self.width = vehicle_data.vehicle_data.width
        self.delta_max = vehicle_data.max_steering_angle()
        self.margin = self.width / 2
        self.min_velocity = velocity_limits["min"]
        self.max_velocity = velocity_limits["max"]
        min_u = np.array([self.min_velocity, -np.tan(self.delta_max) / self.length])
        max_u = np.array([self.max_velocity, np.tan(self.delta_max) / self.length])
        self.min_u, self.max_u = min_u, max_u

    def t2s(self, reference_waypoint: np.array, reference_state: np.array) -> np.array:
        """
        Convert spatial state to temporal state. Either convert self.spatial_
        state with current waypoint as reference or provide reference waypoint
        and reference_state.
        :return Spatial State equivalent to reference state
        """
        ref_x, ref_y, ref_psi = reference_waypoint
        x, y, psi = reference_state
        # Compute spatial state variables=
        e_y = np.cos(ref_psi) * (y - ref_y) - np.sin(ref_psi) * (x - ref_x)
        e_psi = psi - ref_psi
        # Ensure e_psi is kept within range (-pi, pi]
        e_psi = np.mod(e_psi + math.pi, 2 * math.pi) - math.pi
        # time state can be set to zero since it's only relevant for the MPC
        # prediction horizon
        t = 0.0
        return np.array([e_y, e_psi, t])

    def s2t(
        self,
        reference_waypoints: ReferencePath,
        reference_states: np.array,
    ) -> np.array:
        """
        Convert spatial state to temporal state given a reference waypoint.
        :param reference_waypoint: waypoint object to use as reference
        :param reference_state: state vector as np.array to use as reference
        :return Temporal State equivalent to reference state
        """

        # Compute temporal state variables
        xs = reference_waypoints.xs - reference_states[:, 0] * np.sin(
            reference_waypoints.psis
        )
        ys = reference_waypoints.ys + reference_states[:, 0] * np.cos(
            reference_waypoints.psis
        )
        psis = reference_waypoints.psis + reference_states[:, 1]

        return np.array([xs, ys, psis])

    def linearise(self, reference_path: ReferencePath) -> Tuple[np.array]:
        """
        Linearise the system equations around provided reference values.
        """
        delta_s = reference_path.distances
        kappa_ref = reference_path.kappas
        v_ref = reference_path.velocities
        n = len(reference_path)
        ones_col = np.ones(n)
        zeros_col = np.zeros(n)

        ###################
        # System Matrices #
        ###################
        # Construct Jacobian Matrices
        A = np.zeros((n, 3, 3))
        a_1 = np.vstack([ones_col, delta_s, zeros_col]).T
        a_2 = np.vstack([-(kappa_ref**2) * delta_s, ones_col, zeros_col]).T
        a_3 = np.vstack([-kappa_ref / v_ref * delta_s, zeros_col, ones_col]).T
        A[:, 0, :] = a_1
        A[:, 1, :] = a_2
        A[:, 2, :] = a_3

        B = np.zeros((n, 3, 2))
        b_1 = np.zeros((n, 2))
        b_2 = np.zeros((n, 2))
        b_3 = np.zeros((n, 2))
        b_2[:, 1] = delta_s
        b_3[:, 0] = -1 / (v_ref**2) * delta_s
        B[:, 0, :] = b_1
        B[:, 1, :] = b_2
        B[:, 2, :] = b_3

        f = np.zeros((n, 3))
        f[:, 2] = 1 / v_ref * delta_s

        return f, A, B

    def _linearize(self, v_ref, kappa_ref, delta_s):
        """
        Linearize the system equations around provided reference values.
        :param v_ref: velocity reference around which to linearize
        :param kappa_ref: kappa of waypoint around which to linearize
        :param delta_s: distance between current waypoint and next waypoint
        """

        ###################
        # System Matrices #
        ###################
        # Construct Jacobian Matrix
        a_1 = np.array([1, delta_s, 0])
        a_2 = np.array([-(kappa_ref**2) * delta_s, 1, 0])
        a_3 = np.array([-kappa_ref / v_ref * delta_s, 0, 1])

        b_1 = np.array([0, 0])
        b_2 = np.array([0, delta_s])
        b_3 = np.array([-1 / (v_ref**2) * delta_s, 0])

        f = np.array([0.0, 0.0, 1 / v_ref * delta_s])

        A = np.stack((a_1, a_2, a_3), axis=0)
        B = np.stack((b_1, b_2, b_3), axis=0)

        return f, A, B
