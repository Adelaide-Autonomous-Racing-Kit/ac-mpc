import numpy as np


class ReferencePath:
    def __init__(self, n_positions: int):
        self._n_positions = n_positions
        self._reference_path = np.zeros((7, n_positions))

    @property
    def xs(self) -> np.array:
        return self._reference_path[0, :]

    @xs.setter
    def xs(self, xs: np.array):
        self._reference_path[0, :] = xs

    @property
    def ys(self) -> np.array:
        return self._reference_path[1, :]

    @ys.setter
    def ys(self, ys: np.array) -> np.array:
        self._reference_path[1, :] = ys

    @property
    def psis(self) -> np.array:
        return self._reference_path[2, :]

    @psis.setter
    def psis(self, psis: np.array):
        self._reference_path[2, :] = psis

    @property
    def kappas(self) -> np.array:
        return self._reference_path[3, :]

    @kappas.setter
    def kappas(self, kappas: np.array):
        self._reference_path[3, :] = kappas

    @property
    def distances(self) -> np.array:
        return self._reference_path[4, :]

    @distances.setter
    def distances(self, distances: np.array):
        self._reference_path[4, :] = distances

    @property
    def widths(self) -> np.array:
        return self._reference_path[5, :]

    @widths.setter
    def widths(self, widths: np.array):
        self._reference_path[5, :] = widths

    @property
    def velocities(self) -> np.array:
        return self._reference_path[6, :]

    @velocities.setter
    def velocities(self, velocities: np.array):
        self._reference_path[6, :] = velocities

    def __len__(self) -> int:
        return self._n_positions

    def get_state(self, index: int) -> np.array:
        """
        [x, y, psi]
        """
        return self._reference_path[:3, index]
