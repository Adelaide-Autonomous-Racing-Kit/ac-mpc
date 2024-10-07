import unittest
from unittest.mock import Mock

from control.commands import TemporalCommandInterpolator
import numpy as np


class TestTemporalCommandInterpolator(unittest.TestCase):
    def setUp(self):
        mpc = Mock(cum_time=np.array([]), projected_control=np.array([]))
        self._interpolator = TemporalCommandInterpolator(mpc)

    def test_get_closet_command_index(self):
        self._interpolator._MPC.cum_time = np.round(np.linspace(0, 1, 10), 1)
        elapsed_times = [0, 0.22, 1.0, 0.95, 0.77]
        expectations = [(0, 0.0), (2, -0.02), (9, 0.0), (8, -0.05), (7, 0.03)]

        for elapsed_time, expected in zip(elapsed_times, expectations):
            command_index, distance = self._interpolator._get_closet_command_index(
                elapsed_time
            )
            self.assertEqual(expected[0], command_index)
            self.assertAlmostEqual(expected[1], distance)

    def test_interpolate_command(self):
        self._interpolator._MPC.cum_time = np.linspace(0, 1, 11)
        # Command = [velocity, yaw]
        self._interpolator._MPC.projected_control = np.array(
            [
                [17.0, -0.03],
                [0.0, 0.0],
                [5.0, 0.15],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [-5, -0.06],
                [12.0, 0.04],
                [1.0, 0.4],
                [-2.0, 0.02],
            ]
        ).T
        elapsed_times = [-0.1, 0.22, 1.0, 0.95, 0.77, 1.1]
        expected_commands = np.array(
            [
                [17, -0.03],
                [4.2, 0.12],
                [-2.0, 0.02],
                [-0.5, 0.21],
                [6.9, 0.01],
                [-2.0, 0.02],
            ]
        )
        for elapsed_time, expected_command in zip(elapsed_times, expected_commands):
            commands = self._interpolator.get_command(elapsed_time)
            for command, expected in zip(commands, expected_command):
                self.assertAlmostEqual(expected, command)


if __name__ == "__main__":
    unittest.main()
