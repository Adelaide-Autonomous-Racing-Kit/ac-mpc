from __future__ import annotations

from types import SimpleNamespace
from typing import Dict

import numpy as np
import osqp
from scipy import sparse


class ControlSolver:
    def __init__(self, config: Dict, model: SpatialBicycleModel):
        self.__setup(config, model)

    def solve(
        self,
        spatial_state: np.array,
        reference_path: ReferencePath,
    ) -> SimpleNameSpace:
        self._update_references(reference_path)
        self._update_inequalities()
        self._update_problem_bounds(spatial_state, reference_path)
        self._update_costs()
        return self._solve_QP_problem()

    def _update_references(self, reference_path: ReferencePath):
        references = np.array([reference_path.velocities, reference_path.kappas])
        self._urs = np.ravel(references, order="F")
        f, A, B = self._dynamics_model.linearise(reference_path)
        self._f, self._A_block, self._B_block = f, A, B
        self._uq = np.ravel(
            np.einsum("BNi,iB ->BN", self._B_block, references) - self._f
        )

    def _update_inequalities(self):
        # Format matrices
        A = sparse.block_diag(self._A_block, format="csc")
        A = sparse.block_array([[np.zeros((3, A.shape[1]))], [A]], format="csc")
        A = sparse.block_array([[A, np.zeros((A.shape[0], 3))]], format="csc")
        B = sparse.block_diag(self._B_block, format="csc")
        B = sparse.block_array([[np.zeros((3, B.shape[1]))], [B]], format="csc")
        A_x = self._A_1 + sparse.csc_matrix(A)
        B_u = sparse.csc_matrix(B)
        A_eq = sparse.hstack([A_x, B_u])
        self._A = sparse.vstack([A_eq, self._A_ineq], format="csc")

    def _update_problem_bounds(
        self,
        spatial_state: np.array,
        reference_path: ReferencePath,
    ):
        nx = self._n_temporal_states
        widths = reference_path.widths
        # Update dynamic state constraints
        self._x_mins[0] = spatial_state[0]
        self._x_maxs[0] = spatial_state[0]
        x_lower_bounds = (-widths / 2) + self._margin
        x_upper_bounds = (widths / 2) - self._margin
        self._x_mins[nx::nx] = x_lower_bounds
        self._x_maxs[nx::nx] = x_upper_bounds
        # Set reference for state as center-line of drivable area
        self._xrs[nx::nx] = (x_lower_bounds + x_upper_bounds) / 2
        # Get upper and lower bound vectors for equality constraints
        lineq = np.hstack([self._x_mins, self._u_mins])
        uineq = np.hstack([self._x_maxs, self._u_maxs])
        # Get upper and lower bound vectors for inequality constraints
        leq = np.hstack([-spatial_state, self._uq])
        ueq = leq
        self._lower_bounds = np.hstack([leq, lineq])
        self._upper_bounds = np.hstack([ueq, uineq])

    def _update_costs(self):
        self._q = np.hstack(
            [
                -self._q_Q * self._xrs[: -self._n_temporal_states],
                -self._QN.dot(self._xrs[-self._n_temporal_states :]),
                -self._q_R * self._urs,
            ]
        )

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

    def __setup(self, config: Dict, model: SpatialBicycleModel):
        self._dynamics_model = model
        self._unpack_config(config)
        self._allocate_static_matrices(config)
        self._problem = None

    def _unpack_config(self, config: Dict):
        self._n_horizon = config["horizon"] - 1
        self._max_iterations = config["max_iterations"]
        self._n_spatial_states = 2
        self._n_temporal_states = 3
        self._margin = self._dynamics_model.margin

    def _allocate_static_matrices(self, config: Dict):
        n = self._n_horizon
        n_x = self._n_temporal_states
        n_u = self._n_spatial_states
        # Cost Matrices
        self._Q = sparse.diags(config["step_cost"])  # e_y, e_psi, t
        self._R = sparse.diags(config["r_term"])  # velocity, delta
        self._QN = sparse.diags(config["final_cost"])  # e_y, e_psi, t
        # Input constraints
        self._u_max = self._dynamics_model.max_u
        self._u_min = self._dynamics_model.min_u
        # State constraints
        self._x_max = np.array([np.inf, np.inf, np.inf])
        self._x_min = np.array([-np.inf, -np.inf, 0.01])
        # Input Constrains [velocity, angle]
        self._u_maxs = np.kron(np.ones(n), self._u_max)
        self._u_mins = np.kron(np.ones(n), self._u_min)
        self._u_maxs[::2] += 0.1
        self._u_mins[::2] -= 0.1
        # Dynamic state constraints
        self._x_mins = np.kron(np.ones(n + 1), self._x_min)
        self._x_maxs = np.kron(np.ones(n + 1), self._x_max)
        # Reference states
        self._xrs = np.zeros(n_x * (n + 1))
        # Construction matrices
        self._A_1 = sparse.kron(sparse.eye(n + 1), -sparse.eye(n_x))
        self._A_ineq = sparse.eye((n + 1) * n_x + n * n_u)
        self._q_Q = np.tile(np.diag(self._Q.toarray()), n)
        self._q_R = np.tile(np.diag(self._R.toarray()), n)
        # Cost matrix P
        self._P = sparse.block_diag(
            [
                sparse.kron(sparse.eye(n), self._Q),
                self._QN,
                sparse.kron(sparse.eye(n), self._R),
            ],
            format="csc",
        )
