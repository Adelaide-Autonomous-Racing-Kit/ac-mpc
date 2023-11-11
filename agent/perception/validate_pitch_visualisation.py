import copy

import numpy as np
import cv2
from loguru import logger

from utils import CameraInfo

CAMERA_CONFIG = {
    "image_width": 1080,
    "image_height": 540,
    "position": [0.000102, 2.099975, 0.7],
    "pitch_deg": 0.0,
    "vertical_fov_deg": 90,
}

BEV_ROAD = np.zeros([400, 400, 3], np.uint8)
BEV_ROAD[:, 150:200] = (255, 0, 0)
BEV_ROAD[:, 200:250] = (0, 0, 255)

BEV_POINTS = np.array([[150, 0, 1], [250, 0, 1], [150, 400, 1], [200, 400, 1]])

GROUND_POINTS = np.array([[50, 400, 1], [-50, 400, 1], [50, 0, 1], [0, 0, 1]])

thickness = 2
IMAGE_OF_ROAD = np.zeros(
    [CAMERA_CONFIG["image_height"], CAMERA_CONFIG["image_width"], 3], np.uint8
)


track_points = 100
start = 5
finish = 100
width = 10
left_track_wcf = np.array(
    [
        np.ones(track_points) * -width / 2,
        np.linspace(start, finish, track_points),
        np.ones(track_points),
    ]
)
right_track_wcf = np.array(
    [
        np.ones(track_points) * width / 2,
        np.linspace(start, finish, track_points),
        np.ones(track_points),
    ]
)


def test_homography_projection():
    pitch = 0
    homography_w2i = get_w2i_homography_with_pitch(pitch)
    left_track_image = transform_points_wcf_to_image(homography_w2i, left_track_wcf)
    right_track_image = transform_points_wcf_to_image(homography_w2i, right_track_wcf)

    draw_tracks_on_image("original", left_track_image, right_track_image)

    pitch = -10
    homography_w2i = get_w2i_homography_with_pitch(pitch)
    left_track_image = transform_points_wcf_to_image(homography_w2i, left_track_wcf)
    right_track_image = transform_points_wcf_to_image(homography_w2i, right_track_wcf)

    draw_tracks_on_image(
        f"pitch rotation by {pitch} deg", left_track_image, right_track_image
    )

    pitch = 10
    homography_w2i = get_w2i_homography_with_pitch(pitch)

    left_track_image = transform_points_wcf_to_image(homography_w2i, left_track_wcf)
    right_track_image = transform_points_wcf_to_image(homography_w2i, right_track_wcf)

    draw_tracks_on_image(
        f"pitch rotation by {pitch} deg", left_track_image, right_track_image
    )

    cv2.waitKey(0)


def get_w2i_homography_with_pitch(pitch_amount):
    camera_config = copy.deepcopy(CAMERA_CONFIG)
    camera_config["pitch_deg"] += pitch_amount
    camera_info = CameraInfo(camera_config)

    homography_w2i = np.linalg.inv(camera_info.homography_i2w)

    return homography_w2i


def transform_points_wcf_to_image(homography_w2i, wcf_points):
    image_points = homography_w2i @ wcf_points
    image_points = image_points[:2] / image_points[2]

    x_mask = (image_points[0] > 0) & (image_points[0] < CAMERA_CONFIG["image_width"])
    y_mask = (image_points[1] > 0) & (image_points[1] < CAMERA_CONFIG["image_height"])
    mask = x_mask & y_mask

    filtered_image_points = image_points[:, mask]

    return filtered_image_points


def draw_tracks_on_image(window_name, left_track, right_track):
    image_of_road = copy.deepcopy(IMAGE_OF_ROAD)
    cv2.polylines(
        image_of_road, [left_track.astype(np.int32).T], False, (255, 0, 0), thickness
    )
    cv2.polylines(
        image_of_road, [right_track.astype(np.int32).T], False, (0, 0, 255), thickness
    )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image_of_road)


if __name__ == "__main__":
    test_homography_projection()
