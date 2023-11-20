import copy
import unittest

import numpy as np

from commands import TemporalCommandInterpolator


class TestTemporalCommandInterpolator(unittest.TestCase):
    def setUp(self):
        self._interpolator = TemporalCommandInterpolator()

    def test_get_closet_command_index(self):
        self._interpolator._cum_time = np.round(np.linspace(0, 1, 10), 1)
        elapsed_times = [0, 0.22, 1.0, 0.95, 0.77]
        expectations = [(0, 0.0), (2, -0.02), (9, 0.0), (8, -0.05), (7, 0.03)]

        for elapsed_time, expected in zip(elapsed_times, expectations):
            command_index, distance = self._interpolator._get_closet_command_index(
                elapsed_time
            )
            self.assertEqual(expected[0], command_index)
            self.assertAlmostEqual(expected[1], distance)

    def test_interpolate_command(self):
        self._interpolator._cum_time = np.linspace(0, 1, 11)
        # Command = [steering, brake, throttle]
        self._interpolator._commands = np.array(
            [
                [0.5, 0, 0.2],
                [0.0, 0.0, 0.0],
                [-1.0, 0.5, 0.1],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-0.5, 0.5, 0.6],
                [1.0, 0.0, 0.4],
                [1.0, 0.0, 0.4],
                [-1.0, 1.0, 0.2],
            ]
        )
        elapsed_times = [-0.1, 0.22, 1.0, 0.95, 0.77, 1.1]
        expected_commands = np.array(
            [
                [0.5, 0, 0.2],
                [-0.6, 0.6, 0.08],
                [-1.0, 1.0, 0.2],
                [0.0, 0.5, 0.3],
                [0.55, 0.15, 0.46],
                [-1.0, 1.0, 0.2],
            ]
        )
        for elapsed_time, expected_command in zip(elapsed_times, expected_commands):
            commands = self._interpolator.get_command(elapsed_time)
            for command, expected in zip(commands, expected_command):
                self.assertAlmostEqual(expected, command)


if __name__ == "__main__":
    unittest.main()
