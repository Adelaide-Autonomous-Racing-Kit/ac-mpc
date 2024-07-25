import time

import matplotlib.pyplot as plt
import numpy as np

from ace.steering import SteeringGeometry
from control.controller import build_mpc
from control.utils import (
    get_hairpin_track,
    get_chicane_track,
    get_curved_track,
    get_straight_track,
)

from loguru import logger


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
    vehicle_data = SteeringGeometry("data/audi_r8_lms_2016")
    mpc = build_mpc(config, vehicle_data)
    N = config["horizon"]

    road_width = 100
    path_type = "hairpin"  # "hairpin", "curve", "chicane"

    # Experiment settings
    colours = ["b", "c", "k", "g", "m", "y", "r"]
    show_example_by_example = False
    angle = 0.1
    experiments = 7

    # Curve experiments
    quadratic_coeff = 0.02
    curve_coefficient = np.linspace(-quadratic_coeff, quadratic_coeff, experiments)

    # Constant radius experiments
    test_radii = np.linspace(10, 100, experiments)

    # Chicane experiments
    chicane_width = 40
    distance_to_chicane = np.linspace(40, 100, experiments)

    # Straight line experiments
    line_of_sight = np.linspace(40, 200, experiments)

    for path_type in ["hairpin", "chicane", "curve", "straight"]:
        # for path_type in ["curve"]:
        fig, ax = plt.subplots(1)
        fig1, ax1 = plt.subplots(2)
        ax = [ax]

        for i in range(experiments):
            if path_type == "hairpin":
                x, y = get_hairpin_track(test_radii[i], N, -np.pi / 6)
            elif path_type == "chicane":
                x, y = get_chicane_track(
                    distance_to_chicane[i], chicane_width, N, angle
                )
            elif path_type == "curve":
                x, y = get_curved_track(curve_coefficient[i], N, angle)
            elif path_type == "straight":
                x, y = get_straight_track(line_of_sight[i], N, angle)

            test_reference_path = np.stack(
                [
                    x,
                    y,
                    np.ones(N) * road_width,
                ]
            ).T

            st = time.time()
            mpc.get_control(test_reference_path, offset=0.0)
            print(f"Time to solve get_control: {time.time() - st:.4f}")

            # print(controller.current_prediction)
            cum_dist = np.cumsum(mpc.reference_path.distances)
            # print(mpc.current_control)

            if show_example_by_example:
                ax[0].clear()
                ax1[0].clear()
                ax1[1].clear()

            ax[0].scatter(
                test_reference_path[:, 0],
                test_reference_path[:, 1],
                c="g",
                label="reference",
            )

            ax[0].scatter(
                mpc.current_prediction.T[0],
                mpc.current_prediction.T[1],
                c=colours[i],
                label="predicted",
            )
            cum_dist = np.concatenate([[0], cum_dist])
            ax1[0].set_title("Velocity reference: green, Planned velocity: red, m/s")
            ax1[0].plot(mpc.projected_control[0], c=colours[i])
            ax1[0].plot(mpc.speed_profile, "--", c=colours[i])

            ax1[1].set_title("Steering command (rad)")
            ax1[1].plot(mpc.projected_control[1], c=colours[i])

            ax[0].set_aspect(1)

            if show_example_by_example:
                plt.pause(0.5)

        if not show_example_by_example:
            plt.show()
            plt.close()


def get_spatial_MPC_failures():
    failure_reference_paths = [[]]

    return failure_reference_paths


if __name__ == "__main__":
    main()
