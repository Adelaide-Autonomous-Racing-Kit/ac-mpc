import math

import numpy as np
from loguru import logger


class SpatialBicycleModel:
    def __init__(self, n_states, wheel_base, width):
        self.n_states = n_states
        self.length = wheel_base
        self.width = width
        self.safety_margin = width / 2

    def t2s(self, reference_waypoint, reference_state):
        """
        Convert spatial state to temporal state. Either convert self.spatial_
        state with current waypoint as reference or provide reference waypoint
        and reference_state.
        :return Spatial State equivalent to reference state
        """
        ref_x, ref_y, ref_psi = (
            reference_waypoint["x"],
            reference_waypoint["y"],
            reference_waypoint["psi"],
        )
        x, y, psi = reference_state
        # Compute spatial state variables
        if isinstance(reference_state, np.ndarray):
            e_y = np.cos(ref_psi) * (y - ref_y) - np.sin(ref_psi) * (x - ref_x)
            e_psi = psi - ref_psi

            # Ensure e_psi is kept within range (-pi, pi]
            e_psi = np.mod(e_psi + math.pi, 2 * math.pi) - math.pi
        else:
            logger.error("Reference State type not supported!")
            e_y, e_psi = None, None
            exit(1)

        # time state can be set to zero since it's only relevant for the MPC
        # prediction horizon
        t = 0.0

        return [e_y, e_psi, t]

    def s2t(self, reference_waypoint, reference_state):
        """
        Convert spatial state to temporal state given a reference waypoint.
        :param reference_waypoint: waypoint object to use as reference
        :param reference_state: state vector as np.array to use as reference
        :return Temporal State equivalent to reference state
        """

        # Compute temporal state variables
        if isinstance(reference_state, np.ndarray):
            x = reference_waypoint["x"] - reference_state[0] * np.sin(
                reference_waypoint["psi"]
            )
            y = reference_waypoint["y"] + reference_state[0] * np.cos(
                reference_waypoint["psi"]
            )
            psi = reference_waypoint["psi"] + reference_state[1]
        else:
            logger.error("Reference State type not supported!")
            x, y, psi = None, None, None
            exit(1)

        return [x, y, psi]

    def linearize(self, v_ref, kappa_ref, delta_s):
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
        a_2 = np.array([-(kappa_ref ** 2) * delta_s, 1, 0])
        a_3 = np.array([-kappa_ref / v_ref * delta_s, 0, 1])

        b_1 = np.array([0, 0])
        b_2 = np.array([0, delta_s])
        b_3 = np.array([-1 / (v_ref ** 2) * delta_s, 0])

        f = np.array([0.0, 0.0, 1 / v_ref * delta_s])

        A = np.stack((a_1, a_2, a_3), axis=0)
        B = np.stack((b_1, b_2, b_3), axis=0)

        return f, A, B
