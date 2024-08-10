import copy

import cv2
import numpy as np

from ..perception.utils import DUMMY_GROUND_POINTS, CameraInfo

CAMERA_CONFIG = {
    "width": 1920,
    "height": 1080,
    "position": [0.000102, 2.099975, 0.7],
    "rotation_deg": [90.0, 0.0, 0.0],
    "vertical_fov": 60,
}

BEV_ROAD = np.zeros(400, 400, 3)
BEV_ROAD[150:200, :] = (255, 0, 0)
BEV_ROAD[200:250, :] = (0, 0, 255)


# def test_homography_consistency()

# def test_overhead_camera_reprojection_calculations()


def test_yaw_rotation():
    camera_config = copy.copy(CAMERA_CONFIG)
    camera_config["rotation"][2] = 45
    camera_info = CameraInfo(camera_config)

    homography_w2i = np.linalg.inv(camera_info.homography_i2w)

    warped_BEV = cv2.warpPerspective(BEV_ROAD, homography_w2i)

    cv2.namedWindow("warped_bev", cv2.WINDOW_NORMAL)
    cv2.imshow("warped_bev", warped_BEV)
    cv2.waitKey(0)


# def test_pitch_rotation():


# def test_roll_rotation():


def test_image_to_ground_projection():
    camera_info = CameraInfo(CAMERA_CONFIG)
    image_points = camera_info._get_corresponding_image_point(DUMMY_GROUND_POINTS)

    projected_ground_points = np.matmul(camera_info.homography_i2w, image_points.T)
    projected_ground_points /= projected_ground_points[2]

    assert np.all(
        np.isclose(projected_ground_points, DUMMY_GROUND_POINTS)
    ), "Homography "


if __name__ == "__main__":
    test_yaw_rotation()
