from __future__ import annotations

"""
Regression tests for the spatial MPC QP state-transition dynamics.
"""

import numpy as np

from acmpc.control.dynamics import SpatialBicycleModel
from acmpc.control.spatial_mpc import Controls, SpatialMPC

CONFIG = {
    "horizon": 50,
    "speed_profile_constraints": {
        "v_min": 0.5,
        "v_max": 84.0,
        "a_min": -12.0,
        "a_max": 8.0,
        "ay_max": 16.0,
        "ki_min": 0.005,
        "end_velocity": 20.0,
    },
    "step_cost": [10.0, 20.0, 1.0],  # e_y, e_psi, t
    "r_term": [0.1, 10.0],  # velocity, steering
    "s_term": [1.0e-1, 10000.0],  # velocity, steering
    "final_cost": [10.0, 20.0, 1.0],  # e_y, e_psi, t
}
N = CONFIG["horizon"]
ROAD_WIDTH = 100.0


class _StubVehicleData:
    wheelbase = 2.63
    width = 1.94


class _StubVehicle:
    vehicle_data = _StubVehicleData()

    def max_steering_angle(self):
        return np.deg2rad(38.0)


def _build_mpc() -> tuple[SpatialMPC, SpatialBicycleModel]:
    model = SpatialBicycleModel(_StubVehicle(), {"max": 84.0, "min": 0.5})
    return SpatialMPC(CONFIG, model), model


def _build_arc_path(radius: float, theta_end: float) -> np.array:
    theta = np.linspace(0.0, theta_end, N)
    x = radius * (1.0 - np.cos(theta))
    y = radius * np.sin(theta)
    return np.stack([x, y, np.ones(N) * ROAD_WIDTH]).T


def _solve_once(mpc: SpatialMPC, path: np.array):
    reference_path = mpc.construct_waypoints(path)
    reference_path = mpc.compute_speed_profile(
        reference_path,
        end_vel=CONFIG["speed_profile_constraints"]["end_velocity"],
        start_vel=2.0,
    )
    state = np.array([0.0, 0.0, np.pi / 2])
    spatial_state = mpc.model.t2s(reference_path.get_state(0), state)
    previous_controls = Controls(reference_path.velocities, reference_path.kappas)
    dec = mpc._control_solver.solve(
        spatial_state,
        reference_path,
        previous_controls,
    )
    return reference_path, dec


def test_implied_transition_matches_thesis():
    mpc, _ = _build_mpc()
    reference_path, _ = _solve_once(mpc, _build_arc_path(25.0, 1.0))

    solver = mpc._control_solver
    n = solver._n_horizon
    ds = reference_path.distances
    kappa_r = reference_path.kappas
    v_ref = reference_path.velocities
    assert len(v_ref) == n

    for k in range(n):
        row = 3 * (k + 1)
        rhs = solver._lower_bounds[row : row + 3]
        assert np.isclose(rhs[0], 0.0, atol=1e-12), (k, rhs)
        assert np.isclose(rhs[1], ds[k] * kappa_r[k], atol=1e-12), (k, rhs[1])
        assert np.isclose(rhs[2], -2.0 * ds[k] / v_ref[k], atol=1e-9), (k, rhs[2])


def test_planned_kappa_follows_path_kappa():
    mpc, _ = _build_mpc()
    reference_path, dec = _solve_once(mpc, _build_arc_path(25.0, 1.0))
    assert dec.info.status in [
        "solved",
        "solved inaccurate",
        "maximum iterations reached",
    ], dec.info

    control_signals = np.array(dec.x[-(N - 1) * 2 :])
    kappa_solution = control_signals[1::2]
    ratio = kappa_solution[5:] / reference_path.kappas[5:]
    assert np.all(np.abs(ratio - 1.0) < 0.1), f"kappa ratio: {ratio}"


def test_qp_is_primal_feasible():
    mpc, _ = _build_mpc()
    _, dec = _solve_once(mpc, _build_arc_path(25.0, 1.0))
    assert dec.info.status in [
        "solved",
        "solved inaccurate",
        "maximum iterations reached",
    ], dec.info

    solver = mpc._control_solver
    a_z = solver._A @ dec.x
    violation = np.maximum(
        np.maximum(solver._lower_bounds - a_z, a_z - solver._upper_bounds), 0.0
    )
    assert violation.max() < 1e-5, (
        f"max bound violation {violation.max():.3e} at row {int(violation.argmax())}"
    )


def test_start_pin_respects_grip_limit():
    """
    v_max[0] is capped by the curvature limit, not replaced by start_velocity**2.
    """
    mpc, _ = _build_mpc()
    reference_path = mpc.construct_waypoints(_build_arc_path(25.0, 1.0))
    dec = mpc._speed_profile_solver.solve(
        reference_path, end_velocity=20.0, start_velocity=30.0
    )
    assert dec.info.status in [
        "solved",
        "maximum iterations reached",
        "solved inaccurate",
    ], dec.info

    v = np.sqrt(dec.x)
    v_max_grip = np.sqrt(
        mpc.speed_profile_constraints["ay_max"] / np.abs(reference_path.kappas) + 2.0
    )
    assert np.all(v <= v_max_grip + 1e-6), (v.max(), v_max_grip.min())


def main():
    tests = [
        test_implied_transition_matches_thesis,
        test_planned_kappa_follows_path_kappa,
        test_qp_is_primal_feasible,
        test_start_pin_respects_grip_limit,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
