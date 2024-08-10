import copy
import unittest

import numpy as np
from utils import CameraInfo


class TestCamera(unittest.TestCase):
    def setUp(self):
        self.homogeneous_wcf_points = np.array(
            [[-5, 10, 1], [5, 10, 1], [-5, 50, 1], [5, 50, 1]]
        ).T

        self.camera_config = {
            "image_width": 1080,
            "image_height": 540,
            "position": [0.0, 0.0, 1.0],
            "pitch_deg": 0.0,
            "vertical_fov_deg": 60,
        }

    def test_focal_length_calculation(self):
        camera_config = copy.deepcopy(self.camera_config)
        camera_config["image_height"] = 540
        camera_config["image_width"] = 540
        camera_config["vertical_fov_deg"] = 90

        camera_info = CameraInfo(camera_config)

        self.assertAlmostEqual(camera_info.focal_length, camera_info.height / 2)

    def test_camera_matrix_calculation(self):
        camera_config = copy.deepcopy(self.camera_config)
        camera_config["image_height"] = 540
        camera_config["image_width"] = 1080
        camera_config["vertical_fov_deg"] = 90

        camera_info = CameraInfo(camera_config)

        expected_camera_matrix = np.array([[270, 0, 540], [0, 270, 270], [0, 0, 1]])

        self.assertTrue(
            np.all(np.isclose(camera_info.camera_matrix, expected_camera_matrix))
        )

    def test_rotation_matrices(self):
        camera_config = copy.deepcopy(self.camera_config)
        camera_config["pitch_deg"] = 0.0
        camera_info = CameraInfo(camera_config)

        expected_rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        self.assertTrue(
            np.all(np.isclose(camera_info.rotation_matrix, expected_rotation_matrix)),
            "no pitch rotation failed",
        )

        camera_config = copy.deepcopy(self.camera_config)
        camera_config["pitch_deg"] = 10.0
        camera_info = CameraInfo(camera_config)

        expected_rotation_matrix = np.array(
            [[1, 0, 0], [0, -0.173648, -0.984808], [0, 0.984808, -0.173648]]
        )

        self.assertTrue(
            np.all(np.isclose(camera_info.rotation_matrix, expected_rotation_matrix)),
            "positive pitch rotation failed",
        )

        camera_config = copy.deepcopy(self.camera_config)
        camera_config["pitch_deg"] = -10.0
        camera_info = CameraInfo(camera_config)

        expected_rotation_matrix = np.array(
            [[1, 0, 0], [0, 0.173648, -0.984808], [0, 0.984808, 0.173648]]
        )

        self.assertTrue(
            np.all(np.isclose(camera_info.rotation_matrix, expected_rotation_matrix)),
            "negative pitch rotation failed",
        )

    def test_extrinsics(self):
        world_coordinate_frame_points = np.array(
            [[50, 400, 0], [-50, 400, 0], [50, 0, 0], [0, 0, 0]]
        )

        camera_config = copy.deepcopy(self.camera_config)
        camera_config["position"] = [0.0, 0.0, 1.0]
        camera_info = CameraInfo(camera_config)

        ccf_points = camera_info.translate_points_from_world_to_camera_frame(
            world_coordinate_frame_points
        )

        expected_ccf_points = np.array(
            [[50, 1.0, 400], [-50, 1.0, 400], [50, 1.0, 0], [0, 1.0, 0]]
        )

        self.assertTrue(
            np.all(np.isclose(ccf_points, expected_ccf_points)),
            "wcf to ccf rotation did not work",
        )

        camera_config = copy.deepcopy(self.camera_config)
        camera_config["position"] = [0.0, 2.0, 1.0]
        camera_info = CameraInfo(camera_config)

        ccf_points = camera_info.translate_points_from_world_to_camera_frame(
            world_coordinate_frame_points
        )

        expected_ccf_points = np.array(
            [[50, 1.0, 400 - 2], [-50, 1.0, 400 - 2], [50, 1.0, -2], [0, 1.0, -2]]
        )

        self.assertTrue(
            np.all(np.isclose(ccf_points, expected_ccf_points)),
            "wcf to ccf rotation",
        )

        camera_config = copy.deepcopy(self.camera_config)
        camera_config["position"] = [0.0, 0.0, 1.0]
        camera_config["pitch_deg"] = 45.0
        camera_info = CameraInfo(camera_config)

        ccf_points = camera_info.translate_points_from_world_to_camera_frame(
            world_coordinate_frame_points
        )

        z_diff = np.sin(45 * np.pi / 180) * 400
        y_diff = np.sin(45 * np.pi / 180) * 1.0

        expected_ccf_points = np.array(
            [
                [50, -z_diff + y_diff, z_diff + y_diff],
                [-50, -z_diff + y_diff, z_diff + y_diff],
                [50, y_diff, y_diff],
                [0, y_diff, y_diff],
            ]
        )

        self.assertTrue(
            np.all(np.isclose(ccf_points, expected_ccf_points)),
            "wcf to ccf rotation",
        )

    def test_intrinsics(self):
        camera_points = np.array([[0, 0, 20], [2, 0, 1], [0, 1, 1], [2, 1, 1]]).T

        camera_config = copy.deepcopy(self.camera_config)
        camera_config["vertical_fov_deg"] = 90
        camera_config["image_width"] = 1000
        camera_config["image_height"] = 500
        camera_info = CameraInfo(camera_config)

        image_points = camera_info.translate_points_from_camera_to_image_frame(
            camera_points
        )

        expected_image_points = np.array(
            [
                [camera_info.width / 2, camera_info.height / 2],
                [camera_info.width, camera_info.height / 2],
                [camera_info.width / 2, camera_info.height],
                [camera_info.width, camera_info.height],
            ]
        )

        self.assertTrue(
            np.all(np.isclose(image_points, expected_image_points)),
            "camera coordinate frame to image translation is wrong",
        )

    def test_world_to_image_point_conversions(self):
        world_points = np.array([[0, 1, 1], [2, 1, 1], [0, 1, 0], [2, 1, 0]])

        camera_config = copy.deepcopy(self.camera_config)
        camera_config["vertical_fov_deg"] = 90
        camera_config["image_width"] = 1000
        camera_config["image_height"] = 500
        camera_info = CameraInfo(camera_config)

        image_points = camera_info.translate_points_from_world_to_image_frame(
            world_points
        )

        expected_image_points = np.array(
            [
                [camera_info.width / 2, camera_info.height / 2],
                [camera_info.width, camera_info.height / 2],
                [camera_info.width / 2, camera_info.height],
                [camera_info.width, camera_info.height],
            ]
        )

        self.assertTrue(
            np.all(np.isclose(image_points, expected_image_points)),
            f"world coordinate frame to image translation is wrong \n Expected\n {expected_image_points} \n Calculated\n {image_points}",
        )

        world_points_on_the_ground = np.array(
            [[-10, 400, 0], [10, 400, 0], [-10, 40, 0], [10, 40, 0]]
        )
        camera_config = copy.deepcopy(self.camera_config)
        camera_config["vertical_fov_deg"] = 90
        camera_config["image_width"] = 1000
        camera_config["image_height"] = 500
        camera_info = CameraInfo(camera_config)

        image_points_homo = camera_info.translate_points_from_ground_to_image_plane(
            world_points_on_the_ground[:, :2]
        )
        image_points = camera_info.translate_points_from_world_to_image_frame(
            world_points_on_the_ground
        )

        self.assertTrue(
            np.all(np.isclose(image_points_homo, image_points)),
            f"Homography projection to the image plane is not the same as using extrinsics",
        )

        ground_points_homo = camera_info.translate_points_from_image_to_ground_plane(
            image_points_homo
        )

        self.assertTrue(
            np.all(np.isclose(ground_points_homo, world_points_on_the_ground[:, :2])),
            f"Homography projection to the ground plane is not the same as the inverse",
        )


if __name__ == "__main__":
    unittest.main()
