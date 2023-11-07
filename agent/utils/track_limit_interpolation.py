from typing import List
import numpy as np

MINIMUM_N_OBS = 5
TRACK_WIDTH = 9.0


def maybe_interpolate_track_limit(track_limits: List[np.array]):
    if track_limits[0].shape[0] < MINIMUM_N_OBS:
        # interpolate left from right
        track_limits[0] = interpolate_left_track_limit(track_limits[1])
    if track_limits[1].shape[0] < MINIMUM_N_OBS:
        # interpolate right from left
        track_limits[1] = interpolate_right_track_limit(track_limits[0])


def interpolate_left_track_limit(right_track_limit: np.array) -> np.array:
    return interpolate_track_limit(right_track_limit, False)


def interpolate_right_track_limit(left_track_limit: np.array) -> np.array:
    return interpolate_track_limit(left_track_limit, True)


def interpolate_track_limit(
    track_limit: np.array, is_interpolating_right: bool
) -> np.array:
    i = 1
    interpolated_track_limit = np.zeros((track_limit.shape))[1:-1]
    for track_point in track_limit[1:-1]:
        ahead_point = track_limit[i + 1]
        behind_point = track_limit[i - 1]
        line = ahead_point - behind_point
        unit_normal = np.array([-line[1], line[0]]) / np.linalg.norm(line)
        if is_interpolating_right:
            interpolated_track_limit[i - 1] = track_point - unit_normal * 9.0
        else:
            interpolated_track_limit[i - 1] = track_point + unit_normal * 9.0
        i += 1
    return interpolated_track_limit
