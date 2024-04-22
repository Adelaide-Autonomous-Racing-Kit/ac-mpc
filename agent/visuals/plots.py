from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from visuals import utils

N_POINTS = 400


def _get_transformed_points(track: np.array, index: int, state: np.array) -> np.array:
    indices = np.arange(index, index + N_POINTS)
    track = utils.get_track_points_by_index(indices, track)
    angle = -state[2] + np.pi / 2
    map_rot = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return utils.transform_track_points(track, state[:2], map_rot)


def draw_localisation_map(agent: ElTuarMPC, canvas: np.array) -> np.array:
    localiser = agent.localiser
    if not (localiser and localiser.is_localised):
        return canvas
    state, i_centre, i_left, i_right = localiser.visualisation_estimated_position

    centre_track = _get_transformed_points(localiser.centre_track, i_centre, state)
    left_track = _get_transformed_points(localiser.left_track, i_left, state)
    right_track = _get_transformed_points(localiser.right_track, i_right, state)

    """
    left_track1 = smooth_track_with_polyfit(
        left_track.T, number_of_track_points, degree=4
    )
    right_track1 = smooth_track_with_polyfit(
        right_track.T, number_of_track_points, degree=4
    )
    """

    logger.debug(f"Right: {right_track}")
    logger.debug(f"Left: {left_track}")
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
        # logger.info("Original Center track In plot")
        # utils.draw_track_lines_on_bev(
        #    canvas, 4, [agent.original_centre_track], colour=(255, 0, 255)
        # )
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


def draw_segmentation_map(agent: ElTuarMPC, canvas: np.array) -> np.array:
    return np.transpose(agent.perception.output_mask, axes=(1, 2, 0)) * 255


# V3 drivable FPN
COLOUR_LIST = np.array(
    [
        (0, 0, 0),
        (0, 255, 249),
        (84, 84, 84),
        (255, 119, 51),
        (255, 255, 255),
        (255, 255, 0),
        (170, 255, 128),
        (255, 42, 0),
        (153, 153, 255),
        (255, 179, 204),
    ]
)


def draw_visualised_predictions(agent: ElTuarMPC, canvas: np.array) -> np.array:
    vis = agent.perception.output_visualisation
    vis = np.squeeze(np.array(COLOUR_LIST[vis], dtype=np.uint8))
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def draw_camera_feed(agent: ElTuarMPC, canvas: np.array) -> np.array:
    return cv2.cvtColor(agent.perception.input_image, cv2.COLOR_BGR2RGB)
