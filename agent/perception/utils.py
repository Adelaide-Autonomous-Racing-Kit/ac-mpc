import math

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import UnivariateSpline


class CameraInfo:
    def __init__(self, camera_config):
        self.width = camera_config["image_width"]
        self.height = camera_config["image_height"]
        self.position = camera_config["position"]
        self.pitch_rotation = camera_config["pitch_deg"]
        self.vertical_fov_deg = camera_config["vertical_fov_deg"]

        self.validate_camera_height()

        self._init_focal_length()
        self._init_camera_matrix()
        self._init_rotation_matrix()
        self._init_translation_matrix()
        self._init_extrinsic_matrix()
        self._init_full_camera_transformation_matrix()
        self._init_homographies()

    def validate_camera_height(self):
        assert self.position[2] > 0.0, "You cannot have a camera below the ground"

    def _init_focal_length(self):
        self.focal_length = self.height / (
            2 * math.tan(math.radians(self.vertical_fov_deg) / 2)
        )

    def _init_camera_matrix(self):
        camera_matrix = np.float32(
            [
                [self.focal_length, 0, self.width / 2],
                [0, self.focal_length, self.height / 2],
                [0, 0, 1],
            ]
        )
        self.camera_matrix = camera_matrix

    def _init_rotation_matrix(self):
        rotation_matrix = R.from_euler(
            "x", 90 + self.pitch_rotation, degrees=True
        ).as_matrix()
        self.rotation_matrix = rotation_matrix

    def _init_translation_matrix(self):
        translation_matrix = -np.array(self.position).astype(np.float32)
        self.translation_matrix = translation_matrix

    def _init_extrinsic_matrix(self):
        translation_matrix = self.translation_matrix.reshape(-1, 1)
        translation_matrix = self.rotation_matrix @ translation_matrix

        self.extrinsic_calibration = np.hstack(
            [self.rotation_matrix, translation_matrix]
        )

    def _init_full_camera_transformation_matrix(self):
        self.full_camera_transformation_matrix = (
            self.camera_matrix @ self.extrinsic_calibration
        )

    def _init_homographies(self):
        self.homography_w2i = self.full_camera_transformation_matrix[:, [0, 1, 3]]
        self.homography_i2w = np.linalg.inv(self.homography_w2i)

    def translate_points_from_world_to_camera_frame(self, world_points):
        world_points = self.make_homogeneous(world_points).T
        camera_frame_points = self.extrinsic_calibration @ world_points

        return camera_frame_points.T

    def translate_points_from_camera_to_image_frame(self, camera_points):
        camera_points = np.matmul(self.camera_matrix, camera_points)
        camera_points = camera_points[:2] / camera_points[2]

        return camera_points.T

    def translate_points_from_world_to_image_frame(self, world_points):
        world_points = self.make_homogeneous(world_points).T
        image_points = self.full_camera_transformation_matrix @ world_points
        image_points = image_points[:2] / image_points[2]

        return image_points.T

    def translate_points_from_ground_to_image_plane(self, ground_points):
        ground_points = self.make_homogeneous(ground_points).T
        image_points = self.homography_w2i @ ground_points
        image_points = image_points[:2] / image_points[2]
        return image_points.T

    def translate_points_from_image_to_ground_plane(self, image_points):
        image_points = self.make_homogeneous(image_points).T
        ground_points = self.homography_i2w @ image_points
        ground_points = ground_points[:2] / ground_points[2]
        return ground_points.T

    @staticmethod
    def make_homogeneous(points):
        number_of_points = points.shape[0]
        homogeneous_points = np.hstack([points, np.ones((number_of_points, 1))])

        return homogeneous_points


def smooth_track_with_polyfit(track, num_points, degree=3):
    if len(track[:, 0]) == 0:
        xnew = np.linspace(0, 0.1, num_points)
        ynew = np.linspace(0, 2, num_points)
        return np.array([xnew, ynew]).T
    ynew = np.linspace(0, np.max(track[:, 1]), num_points)
    logger.info(f"{track}")
    coeffs = np.polyfit(track[:, 1], track[:, 0], degree)
    xnew = np.polyval(coeffs, ynew)
    return np.array([xnew, ynew]).T


def smooth_track_with_spline(track, num_points, smooth_factor=1e4):
    ynew = np.linspace(0, np.max(track[1]), num_points)
    spl = UnivariateSpline(track[1], track[0])
    spl.set_smoothing_factor(smooth_factor)
    xnew = spl(ynew)
    return np.array([xnew, ynew])
