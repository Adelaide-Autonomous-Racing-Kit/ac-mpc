import math

import cv2
import copy
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import UnivariateSpline

class Camera:

    def __init__(self, CameraInfo):
        pass

    def get_homography(self):
        pass

DUMMY_GROUND_POINTS = np.array([
        [-10, 20, 0],
        [10, 20, 0],
        [-10, 120, 0],
        [10, 120, 0],
    ])

class CameraInfo:

    def __init__(self, camera_info):
        self.width = camera_info["image_width"]
        self.height = camera_info["image_height"]
        # self.colour_channels = camera_info["image_size"][2]
        self.position = camera_info["position"]
        self.rotation_deg = camera_info["rotation_deg"]
        self.vertical_fov_deg = camera_info["vertical_fov_deg"]

        self.focal_length_y = self.get_focal_length_y()
        self.horizontal_fov = self.get_horizontal_fov()
        self.focal_length_x = self.get_focal_length_x()

        self._init_camera_matrix()
        self._init_rotation_matrix()
        self._init_translation_matrix()
        self._init_homography_i2w()

        # self._init_homography()
        
    def get_horizontal_fov(self):
        return math.degrees(2 * math.atan(self.width / self.focal_length_y))

    def get_focal_length_x(self):
        return self.width / (2 * math.tan(math.radians(self.horizontal_fov)))

    def get_focal_length_y(self):
        return self.height / (2 * math.tan(math.radians(self.vertical_fov_deg)/2))

    def _init_camera_matrix(self):
        camera_matrix = np.float32(
            [[self.focal_length_x, 0, self.width / 2],
             [0, self.focal_length_y, self.height / 2],
             [0, 0, 1]]
        )
        self.camera_matrix = camera_matrix
    
    def _init_rotation_matrix(self):
        rotations = np.flip(self.rotation_deg)
        rotation_matrix = R.from_euler("zyx", rotations, degrees=True).as_matrix()
        self.rotation_matrix = rotation_matrix
    
    def _init_translation_matrix(self):
        translation_matrix = -np.array(self.position).astype(np.float32)
        self.translation_matrix = translation_matrix
    
    def _init_homography_i2w(self):
        # TODO: Try to go direct from normal camera transform to homography rather than estimating
        image_points = self._get_corresponding_image_point(DUMMY_GROUND_POINTS)
        
        homogeneous_ground_points = copy.copy(DUMMY_GROUND_POINTS)
        homogeneous_ground_points[:, 2] = 1

        homography = cv2.findHomography(image_points, homogeneous_ground_points)[0]

        self.homography_i2w = homography

    def _get_corresponding_image_point(self, ground_points):
        camera_points = self._translate_points_from_ground_to_camera_frame(ground_points)
        image_points = self._translate_points_from_camera_to_image_frame(camera_points)

        return image_points


    def _translate_points_from_ground_to_camera_frame(self, ground_points):
        camera_frame_points = np.add(ground_points, self.translation_matrix)
        camera_frame_points = np.matmul(self.rotation_matrix, camera_frame_points.T)

        return camera_frame_points

    def _translate_points_from_camera_to_image_frame(self, camera_points):
        camera_points = np.matmul(self.camera_matrix, camera_points).T
        scale_factor = camera_points[:, 2].reshape(-1, 1)
        camera_points = np.divide(camera_points, scale_factor)

        return camera_points

def get_camera_homography(image_size, vertical_fov):
    # TODO: Update position and rotation with car specific values from data gen toolkit
    camera_information = {
        "CameraFront": {
            "position": [0.000102, 2.099975, 0.7],
            "rotation": [90.0, 0.0, 0.0],
            "calibration_pts": [
                [-10, 20, 0],
                [10, 20, 0],
                [-10, 120, 0],
                [10, 120, 0],
            ],
        },
        "CameraLeft": {
            "position": [-0.380003, 1.279986, 0.550007],
            "rotation": [110.000002, 0.0, -50.000092],
            "calibration_pts": [[-4, 5, 0], [-4, 20, 0], [-8, 5, 0], [-8, 20, 0]],
        },
        "CameraRight": {
            "position": [0.380033, 1.290036, 0.550005],
            "rotation": [110.000002, 0.0, 50.000092],
            "calibration_pts": [[4, 5, 0], [4, 20, 0], [8, 5, 0], [8, 20, 0]],
        },
    }
    # height, width = image_size
    # horizontal_fov = calculate_horizontal_fov(vertical_fov, width, height)
    # focal_length_y = height / (2 * math.tan(math.radians(vertical_fov)/2))
    # focal_length_x = width / (2 * math.tan(math.radians(horizontal_fov)/2))
 
    homographies = {}

    for camera, information in camera_information.items():
        camera_matrix = np.float32(
            [[focal_length_x, 0, width / 2],
             [0, focal_length_y, height / 2],
             [0, 0, 1]]
        )
        rotations = np.flip(information["rotation"])
        rotation_matrix = R.from_euler("zyx", rotations, degrees=True).as_matrix()
        translation_matrix = -np.array(information["position"]).astype(np.float32)

        ground_points = np.array(information["calibration_pts"])

        # World coordinates to camera coordinates
        camera_points = np.add(ground_points, translation_matrix)
        camera_points = np.matmul(rotation_matrix, camera_points.T)

        # Camera coordinates to image coordinates
        camera_points = np.matmul(camera_matrix, camera_points).T
        camera_points = np.divide(camera_points, camera_points[:, 2].reshape(-1, 1))

        ground_points[:, 2] = 1

        homography = cv2.findHomography(camera_points, ground_points)[0]

        # Sanity check
        check = np.matmul(homography, camera_points.T)
        check /= check[2]
        assert np.all(
            np.isclose(ground_points.T, check)
        ), "Homography calculation is incorrect"

        homographies[camera] = homography

    return homographies


def calculate_horizontal_fov(
    vertical_fov: float,
    width: int,
    height: int,
) -> float:
    """
    Calculates and returns the camera's horizontal field of view in degrees,
    given the camera's image plane height and width in pixels and vertical
    field of view in degrees.

    :param vertical_fov: Vertical field of view in degrees.
    :type vertical_fov: float
    :param width: Image plane width in pixels.
    :type width: int
    :param height: Image plane height in pixels.
    :type height: int
    :return: Horizontal field of view in degrees.
    :rtype: float
    """
    focal_length = height / math.tan(math.radians(vertical_fov) / 2)
    return math.degrees(2 * math.atan(width / focal_length))

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
 