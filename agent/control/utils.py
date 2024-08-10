import numpy as np


def rotate_track_points(x, y, angle):
    path_rotation = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    return np.matmul(path_rotation, np.stack([x, y]))


def get_hairpin_track(radius, number_of_points, angle=0):
    x = np.cos(np.linspace(0, 3 / 2 * np.pi, number_of_points)) * radius - radius
    y = np.sin(np.linspace(0, 3 / 2 * np.pi, number_of_points)) * radius
    return rotate_track_points(x, y, angle)


def get_curved_track(coeff, number_of_points, angle=0):
    x = np.linspace(0, 100, number_of_points)
    y = coeff * x**2
    return rotate_track_points(x, y, angle)


def get_chicane_track(distance_to_chicane, chicane_width, number_of_points, angle=0):
    y = np.linspace(0, 100, number_of_points)
    x = chicane_width / (1 + np.exp(-0.1 * (y - distance_to_chicane)))
    return rotate_track_points(x, y, angle)


def get_straight_track(length, number_of_points, angle=0):
    x = np.zeros(number_of_points)
    y = np.linspace(0, length, number_of_points)
    return rotate_track_points(x, y, angle)
