import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter

from control.dynamics import SpatialBicycleModel
from control.spatial_mpc import SpatialMPC
from control.utils import (
    get_hairpin_track,
    get_chicane_track,
    get_curved_track,
    get_straight_track,
)


def main():
    N = 50
    Q = sparse.diags([0.001, 0.0, 0.0])  # e_y, e_psi, t
    R = sparse.diags([1e-6, 10])  # velocity, delta
    QN = sparse.diags([0.001, 0.0, 0.01])  # e_y, e_psi, t

    road_width = 1000
    path_type = "hairpin"  # "hairpin", "curve", "chicane"

    v_max = 60.0  # m/s
    wheel_base = 2.898
    width = 2.5
    delta_max = 0.3  # rad original 0.66
    ay_max = 3.0  # m/s^2
    a_min = -16  # m/s^2
    a_max = 6  # m/s^2

    InputConstraints = {
        "umin": np.array([0.0, -np.tan(delta_max) / wheel_base]),
        "umax": np.array([v_max, np.tan(delta_max) / wheel_base]),
    }
    StateConstraints = {
        "xmin": np.array([-np.inf, -np.inf, 0]),
        "xmax": np.array([np.inf, np.inf, np.inf]),
    }

    SpeedProfileConstraints = {
        "a_min": a_min,
        "a_max": a_max,
        "v_min": 0.0,
        "v_max": v_max,
        "ay_max": ay_max,
    }

    model = SpatialBicycleModel(n_states=3, wheel_base=wheel_base, width=width)
    controller = SpatialMPC(
        model, N, Q, R, QN, StateConstraints, InputConstraints, SpeedProfileConstraints
    )

    # Test speed profile calulation

    fig, ax = plt.subplots(1)
    track_dict = np.load(
        "agents/utils/our_racetracks/anglesly2.npy", allow_pickle=True
    ).item()
    left_track = track_dict.get("outside_track")
    right_track = track_dict.get("inside_track")
    centre_track = track_dict.get("centre_track")
    centre_track = np.stack(
        [
            centre_track.T[0],
            centre_track.T[1],
            (np.ones(len(centre_track)) * road_width),
        ]
    ).T
    reference_path = controller.construct_waypoints(centre_track)
    reference_path = controller.compute_speed_profile(
        reference_path, controller.SpeedProfileConstraints
    )

    plot_ref_path = np.array(
        [[val["x"], val["y"], val["v_ref"]] for i, val in enumerate(reference_path)]
    ).T
    plot_ref_path[2] = savgol_filter(plot_ref_path[2], 21, 3)
    ax.scatter(plot_ref_path[0], plot_ref_path[1], c=plot_ref_path[2])
    ax.set_aspect(1)
    print(max(plot_ref_path[2]))
    print(min(plot_ref_path[2]))
    plt.show()

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
            controller = SpatialMPC(
                model,
                N,
                Q,
                R,
                QN,
                StateConstraints,
                InputConstraints,
                SpeedProfileConstraints,
            )

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
            control_output = controller.get_control(test_reference_path, offset=0.0)
            print(f"Time to solve get_control: {time.time() - st:.4f}")

            # print(controller.current_prediction)
            cum_dist = np.cumsum(
                [controller.reference_path[i]["dist_ahead"] for i in range(N - 1)]
            )
            print(controller.current_control)
            print(control_output)

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
                controller.current_prediction[0],
                controller.current_prediction[1],
                c=colours[i],
                label="predicted",
            )
            cum_dist = np.concatenate([[0], cum_dist])
            ax1[0].set_title("Velocity reference: green, Planned velocity: red, m/s")
            ax1[0].plot(controller.projected_control[0], c=colours[i])
            ax1[0].plot(controller.speed_profile, "--", c=colours[i])

            ax1[1].set_title("Steering command (rad)")
            ax1[1].plot(controller.projected_control[1], c=colours[i])

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
