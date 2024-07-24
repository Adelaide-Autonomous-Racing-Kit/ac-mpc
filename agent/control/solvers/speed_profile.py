from __future__ import annotations
from typing import Dict, Union
from types import SimpleNamespace

import osqp
import numpy as np
from scipy import sparse


class SpeedProfileSolver:
    def __init__(self, config: Dict):
        self.__setup(config)

    def solve(
        self,
        reference_path: ReferencePath,
        end_velocity: Union[float, None] = None,
    ):
        self._update_velocity_bounds(reference_path, end_velocity)
        self._update_problem_bounds()
        self._update_inequalities(reference_path)
        self._update_costs()
        return self._solve_QP_problem()

    def _update_velocity_bounds(
        self,
        reference_path: ReferencePath,
        end_velocity: Union[float, None] = None,
    ):
        max_ay = self._max_lateral_acceleration
        # Create matrices
        v_max = np.ones(self._n_horizon) * self._max_velocity
        # Dynamic v_max bases on path curvature
        is_bellow_minimum_kappa = np.abs(reference_path.kappas) < self._min_kappa
        v_max_dyn = np.sqrt(max_ay / (np.abs(reference_path.kappas) + self._eps))
        v_max_dyn[is_bellow_minimum_kappa] = self._max_velocity
        v_mins = np.min([v_max_dyn, v_max], axis=0)
        v_maxs = np.max([self._min_velocities, v_mins], axis=0)
        v_max = v_maxs + 2e0
        # Final velocity
        if end_velocity is not None:
            v_max[-1] = end_velocity
        # Update
        self._max_velocities = v_max

    def _update_problem_bounds(self):
        self._lower_bounds = np.hstack([self._min_accelerations, self._min_velocities])
        self._upper_bounds = np.hstack([self._max_accelerations, self._max_velocities])

    def _update_inequalities(self, reference_path: ReferncePath):
        distances = reference_path.distances
        D1_diagonal = np.array([-1 / (2 * distances[:-1]), 1 / (2 * distances[:-1])])
        shape = [self._n_horizon - 1, self._n_horizon]
        D1 = sparse.diags(D1_diagonal, offsets=[0, 1], shape=shape)
        self._A = sparse.vstack([D1, self._D2], format="csc")

    def _update_costs(self):
        self._q = -1 * self._max_velocities

    def _solve_QP_problem(self) -> SimpleNamespace:
        if self._problem is None:
            self._setup_QP_problem()
        else:
            self._update_QP_problem()
        return self._problem.solve()

    def _setup_QP_problem(self):
        self._problem = osqp.OSQP()
        self._problem.setup(
            P=self._P,
            q=self._q,
            A=self._A,
            l=self._lower_bounds,
            u=self._upper_bounds,
            verbose=False,
            max_iter=self._max_iterations,
        )

    def _update_QP_problem(self):
        self._problem.update(
            Ax=self._A.data,
            q=self._q,
            l=self._lower_bounds,
            u=self._upper_bounds,
        )

    @property
    def _min_acceleration(self) -> float:
        return self._constraints["a_min"]

    @property
    def _max_acceleration(self) -> float:
        return self._constraints["a_max"]

    @property
    def _max_lateral_acceleration(self) -> float:
        return self._constraints["ay_max"]

    @property
    def _max_velocity(self) -> float:
        return self._constraints["v_max"]

    @property
    def _min_velocity(self) -> float:
        return self._constraints["v_min"]

    @property
    def _min_kappa(self) -> float:
        return self._constraints["ki_min"]

    def __setup(self, config: Dict):
        self._unpack_config(config)
        self._allocate_static_matrices()
        self._eps = 1e-12
        self._problem = None

    def _unpack_config(self, config: Dict):
        self._n_horizon = config["control_horizon"]
        self._max_iterations = config["max_iterations"]
        self._constraints = config["constraints"]

    def _allocate_static_matrices(self):
        self._D2 = sparse.eye(self._n_horizon)
        self._P = sparse.eye(self._n_horizon, format="csc")
        self._min_velocities = np.ones(self._n_horizon) * self._min_velocity
        self._min_accelerations = np.ones(self._n_horizon - 1) * self._min_acceleration
        self._max_accelerations = np.ones(self._n_horizon - 1) * self._max_acceleration


class LocalisedSpeedProfileSolver(SpeedProfileSolver):
    def _update_velocity_bounds(
        self,
        reference_path: ReferencePath,
        end_velocity: Union[float, None] = None,
    ):
        self._max_velocities = np.ones(self._n_horizon) * self._max_velocity

    def _update_problem_bounds(self):
        self._upper_bounds = np.hstack([self._max_accelerations, self._max_velocities])

    def _update_costs(self):
        self._q = -1 * self._max_velocities

    def _update_QP_problem(self):
        self._problem.update(Ax=self._A.data, q=self._q, u=self._upper_bounds)

    def _allocate_static_matrices(self):
        super()._allocate_static_matrices()
        self._lower_bounds = np.hstack([self._min_accelerations, self._min_velocities])
