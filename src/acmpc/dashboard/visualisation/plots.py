from __future__ import annotations

import cv2
from acmpc.dashboard.visualisation import utils
import numpy as np

N_POINTS = 400


def _get_transformed_points(track: np.array, index: int, state: np.array) -> np.array:
    indices = np.arange(index, index + N_POINTS)
    track = utils.get_track_points_by_index(indices, track)
    angle = -state[2] + np.pi / 2
    map_rot = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return utils.transform_track_points(track, state[:2], map_rot)


def get_blank_canvas(dimension: int, scale: int) -> np.array:
    bev_size = dimension * scale
    return np.zeros((bev_size, bev_size, 3), dtype=np.uint8)


def draw_localisation_map(agent: ElTuarMPC, canvas: np.array) -> np.array:
    localiser = agent.localiser
    if not (localiser and localiser.is_localised):
        return canvas
    state, i_centre, i_left, i_right = localiser.visualisation_estimated_position

    centre_track = _get_transformed_points(localiser.centre_track, i_centre, state)
    left_track = _get_transformed_points(localiser.left_track, i_left, state)
    right_track = _get_transformed_points(localiser.right_track, i_right, state)
    scale = 4
    utils.draw_track_lines_on_bev(canvas, scale, [centre_track], colour=(0, 255, 0))
    utils.draw_track_lines_on_bev(canvas, scale, [left_track], colour=(255, 0, 0))
    utils.draw_track_lines_on_bev(canvas, scale, [right_track], colour=(0, 0, 255))
    return cv2.flip(canvas, 0)


def draw_control_map(agent: ElTuarMPC, canvas: np.array) -> np.array:
    tracks = agent.perception.visualisation_tracks
    predicted_trajectory = agent.controller.predicted_locations
    x, y = predicted_trajectory[:, 0], predicted_trajectory[:, 1]
    try:
        scale = 16
        utils.draw_track_lines_on_bev(
            canvas, scale, [tracks["centre"]], colour=(0, 255, 0)
        )
        utils.draw_track_lines_on_bev(
            canvas, scale, [tracks["right"]], colour=(0, 0, 255)
        )
        utils.draw_track_lines_on_bev(
            canvas, scale, [tracks["left"]], colour=(255, 0, 0)
        )
        utils.draw_track_lines_on_bev(
            canvas, scale, [np.stack([x, y], axis=0).T], colour=(255, 255, 255)
        )
        canvas = cv2.flip(canvas, 0)
    except Exception as e:
        pass
    return canvas
