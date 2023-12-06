from typing import Dict

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


def draw_localisation_map(agent, canvas: np.array, obs: Dict) -> np.array:
    localiser = agent.localiser
    if not (localiser and localiser.localised):
        return None
    state, centre_idx, left_idx, right_idx = localiser.estimated_position

    centre_track = _get_transformed_points(localiser.centre_track, centre_idx, state)
    left_track = _get_transformed_points(localiser.left_track, left_idx, state)
    right_track = _get_transformed_points(localiser.right_track, right_idx, state)
    
    """
    left_track1 = smooth_track_with_polyfit(
        left_track.T, number_of_track_points, degree=4
    )
    right_track1 = smooth_track_with_polyfit(
        right_track.T, number_of_track_points, degree=4
    )
    """
    lines_to_draw = [
        centre_track.astype(np.int32),
        left_track.astype(np.int32),
        right_track.astype(np.int32),
    ]
    utils.draw_track_lines_on_bev(canvas, 4, lines_to_draw, (255, 255, 255))
    return cv2.flip(canvas, 0)


def draw_control_map(agent, canvas: np.array, obs: Dict) -> np.array:
    try:
        utils.draw_track_lines_on_bev(
            canvas, 4, [agent.centre_track_detections], colour=(0, 255, 0)
        )
        # logger.info("Original Center track In plot")
        # utils.draw_track_lines_on_bev(
        #    canvas, 4, [agent.original_centre_track], colour=(255, 0, 255)
        # )
        utils.draw_track_lines_on_bev(
            canvas, 4, [agent.right_track_detections], colour=(0, 0, 255)
        )
        utils.draw_track_lines_on_bev(
            canvas, 4, [agent.left_track_detections], colour=(255, 0, 0)
        )
        x, y = agent.MPC.current_prediction
        utils.draw_track_lines_on_bev(
            canvas, 4, [np.stack([x, y], axis=0).T], colour=(255, 255, 255)
        )
        canvas = cv2.flip(canvas, 0)
    except Exception as e:
        pass
    return canvas


def draw_segmentation_map(agent, canvas: np.array, obs: Dict) -> np.array:
    image = obs["CameraFrontSegm"]
    if "CameraLeftSegm" in obs:
        image = np.hstack((obs["CameraLeftSegm"], image))
    if "CameraRightSegm" in obs:
        image = np.hstack((image, obs["CameraRightSegm"]))
    return image * 255


def draw_visualised_predictions(agent, canvas: np.array, obs: Dict) -> np.array:
    image = obs["vis"]
    return image


def draw_camera_feed(agent, canvas: np.array, obs: Dict) -> np.array:
    image = obs["CameraFrontRGB"]
    if "CameraLeftRGB" in obs:
        image = np.hstack((obs["CameraLeftRGB"], image))
    if "CameraRightRGB" in obs:
        image = np.hstack((image, obs["CameraRightRGB"]))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
