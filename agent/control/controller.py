import copy
from typing import Dict

import numpy as np
from scipy import sparse

from control.dynamics import SpatialBicycleModel
from control.spatial_mpc import SpatialMPC


def build_mpc(control_cfg: Dict, vehicle_data):
    Q = sparse.diags(control_cfg["step_cost"])  # e_y, e_psi, t
    R = sparse.diags(control_cfg["r_term"])  # velocity, delta
    QN = sparse.diags(control_cfg["final_cost"])  # e_y, e_psi, t
    v_min = control_cfg["speed_profile_constraints"]["v_min"]
    v_max = control_cfg["speed_profile_constraints"]["v_max"]
    wheel_base = vehicle_data.vehicle_data.wheelbase
    width = vehicle_data.vehicle_data.width
    delta_max = vehicle_data.max_steering_angle()

    InputConstraints = {
        "umin": np.array([v_min, -np.tan(delta_max) / wheel_base]),
        "umax": np.array([v_max, np.tan(delta_max) / wheel_base]),
    }
    StateConstraints = {
        "xmin": np.array([-np.inf, -np.inf, -np.inf]),
        "xmax": np.array([np.inf, np.inf, np.inf]),
    }

    model = SpatialBicycleModel(n_states=3, wheel_base=wheel_base, width=width)

    spatial_MPC = SpatialMPC(
        model,
        delta_max,
        control_cfg["horizon"],
        Q,
        R,
        QN,
        StateConstraints,
        InputConstraints,
        copy.copy(control_cfg["speed_profile_constraints"]),
    )
    return spatial_MPC
