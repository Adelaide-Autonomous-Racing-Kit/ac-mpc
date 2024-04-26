from typing import Dict

import numpy as np

MINIMUM_N_OBS = 5
TRACK_WIDTH = 9.0


def maybe_interpolate_track_limit(tracks: Dict):
    if tracks["left"].shape[0] < MINIMUM_N_OBS:
        # interpolate left from right
        tracks["left"] = interpolate_left_track_limit(tracks["left"])
    elif tracks["right"].shape[0] < MINIMUM_N_OBS:
        # interpolate right from left
        tracks["right"] = interpolate_right_track_limit(tracks["left"])


def interpolate_left_track_limit(right_track: np.array) -> np.array:
    return interpolate_track_limit(right_track, False)


def interpolate_right_track_limit(left_track: np.array) -> np.array:
    return interpolate_track_limit(left_track, True)


def interpolate_track_limit(track: np.array, is_interpolating_right: bool) -> np.array:
    interpolated_track_limit = np.zeros_like(track)[:-1]
    diff = np.diff(track, axis=0)
    for i, track_point in enumerate(track[:-1]):
        unit_normal = np.array([-diff[i, 1], diff[i, 0]]) / (
            np.linalg.norm(diff[i]) + 1e-6
        )
        if is_interpolating_right:
            interpolated_track_limit[i] = track_point - unit_normal * 9.0
        else:
            interpolated_track_limit[i] = track_point + unit_normal * 9.0
    return interpolated_track_limit
