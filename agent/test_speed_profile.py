from __future__ import annotations
from typing import Dict
import time

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from ace.steering import SteeringGeometry

from control.controller import build_mpc
from utils import load


def main():
    config = {
        "horizon": 100,
        "unlocalised_max_speed": 28,
        "speed_profile_constraints": {
            "v_min": 12.0,
            "v_max": 84.0,
            "a_min": -1.0,
            "a_max": 1.0,
            "ay_max": 5.5,
            "ki_min": 0.005,
            "end_velocity": 14.0,
        },
        "step_cost": [2.0e-3, 5.0e-2, 0.0],  # e_y, e_psi, t
        "r_term": [1.0e-2, 10.0],  # velocity, steering
        "final_cost": [1.0, 0.0, 0.1],  # e_y, e_psi, t
    }
    ay_max = 7.0
    a_min = -0.135
    track_map = load.track_map("track_maps/monza_verysmooth_2.npy")
    vehicle_data = SteeringGeometry("data/audi_r8_lms_2016")
    mpc = build_mpc(config, vehicle_data)
    centre_track = format_track_map(track_map)
    calculate_speed_profile(a_min, ay_max, centre_track, mpc)


def format_track_map(track_map: Dict) -> np.array:
    centre_track = track_map["centre"]
    road_width = 9.5
    # TODO: James cringe at this line of code
    centre_track = np.stack(
        [
            centre_track.T[0],
            centre_track.T[1],
            (np.ones(len(centre_track)) * road_width),
        ]
    ).T
    return centre_track


def calculate_speed_profile(
    a_min: float,
    ay_max: float,
    centre_track: np.array,
    mpc: SpatialMPC,
):
    reference_path = mpc.construct_waypoints(centre_track)
    reference_path = mpc.compute_map_speed_profile(
        reference_path,
        ay_max=ay_max,
        a_min=a_min,
    )
    plot_ref_path = np.array(
        [reference_path.xs, reference_path.ys, reference_path.velocities]
    )
    plot_speed_profile(plot_ref_path)


def plot_speed_profile(ref_path: np.array):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    sp = ax.scatter(
        ref_path[0, :],
        ref_path[1, :],
        c=ref_path[2, :] * 3.6,
        cmap=plt.get_cmap("plasma"),
        edgecolor="none",
    )
    ax.set_aspect(1)
    plt.gray()
    fig.colorbar(sp)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
