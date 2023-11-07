import glob
import json
import pickle

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


class PersistentMap:
    def __init__(self):
        self.left_track = np.zeros((2, 0))
        self.right_track = np.zeros((2, 0))
        self.centre_track = np.zeros((2, 0))

    def get_track(self):
        return self.left_track, self.right_track, self.centre_track

    def update_map(self, observations):
        """
        Take in observations dictionary
        observations can come from mask output or localisation with their respective keys
        """

    def predict_map(self, control):
        """
        Take in the control input and we need to predict where the new map will be
        and add noise to it
        """

    def step(self, control, observations):
        """
        Step through both update and predictions, return the map
        """


def track_mapping():
    with open("racetracks/thruxton/ThruxtonOfficial.json", "rb") as handle:
        track_dict = json.load(handle)
    with open("agents/utils/SLAM_dataset.pickle", "rb") as handle:
        odometry_dicts = pickle.load(handle)

    for i, odometry in enumerate(odometry_dicts[0:]):
        left_track, right_track = odometry["left_track"], odometry["right_track"]
        control_input = odometry["control"]
        gt_state = odometry["gt_state"]
        dt = odometry["dt"]


def get_camera_homography(image_size):
    camera_information = {
        "CameraFront": {
            "position": [0.000102, 2.099975, 0.7],
            "rotation": [90.0, 0.0, 0.0],
            "calibration_pts": [[-10, 20, 0], [10, 20, 0], [-10, 120, 0], [10, 120, 0]],
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

    homographies = []

    for camera, information in camera_information.items():
        height, width = image_size
        camera_matrix = np.float32([[width / 2, 0, width / 2], [0, width / 2, height / 2], [0, 0, 1]])
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

        scale = 4
        rw_points = np.array([[0, 0, 1], [-50, 50, 1], [0, 100, 1], [50, 50, 1]])
        bev_points = np.array([[50, 100, 1], [0, 50, 1], [50, 0, 1], [100, 50, 1]])
        bev_points[:, :2] = bev_points[:, :2] * scale
        homography2 = cv2.findHomography(rw_points, bev_points)[0]
        homography = np.matmul(homography2, homography)

        homographies.append(homography)

    return homographies


def motion_model(control_input, dt):
    """
    Move the particles based on the control input,
    apply noise to the control input to model uncertainty
    in the movements
    Control input will have velocity
    """
    wheel_base = 3.016

    delta, acceleration, velocity = control_input

    x_dot = np.zeros(3)

    ## w.r.t. rear
    x_dot[0] = velocity * np.cos(np.pi / 2)
    x_dot[1] = velocity * np.sin(np.pi / 2)
    x_dot[2] = velocity * np.tan(delta) / wheel_base

    return dt * x_dot


def rotate_translate_image(image, rotation, dx, dy):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D((200, 400), rotation, 1.0)
    image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    translation_matrix = np.array([[1, 0, dx], [0, 1, dy]]).astype(np.float32)

    image = cv2.warpAffine(image, translation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return image


def track_map_to_bev(track_map, size=(400, 400), scale=4):
    bev = np.zeros((100, 100))

    left_track = np.round(track_map["left"])
    right_track = np.round(track_map["right"])
    left_track[0] = left_track[0] + bev.shape[0] / 2
    right_track[0] = right_track[0] + bev.shape[0] / 2

    for i, row in enumerate(bev):
        left_idx = np.where(left_track[1] == i)[0]
        right_idx = np.where(right_track[1] == i)[0]
        if len(left_idx) and len(right_idx):
            left, right = int(left_track[0, left_idx[0]]), int(right_track[0, right_idx[0]])
            row[left:right] = 1

    bev = cv2.flip(cv2.resize(bev, size), flipCode=0)
    cv2.imshow("Track map", bev)
    return bev


def image_mapping():
    dataset_location = "data/datacollection_persistent_map_localisation/datacollection"
    images = glob.glob(f"{dataset_location}/images/*.png")
    masks = glob.glob(f"{dataset_location}/masks/*.png")
    with open(f"{dataset_location}/commands/commands.json", "rb") as handle:
        commands = json.load(handle)
    H_front, H_left, H_right = get_camera_homography([384, 512])
    size = 400
    persistent_map = np.zeros((size, size))
    map_bev = np.zeros((size, size))
    track_maps = np.load(f"{dataset_location}/maps/maps.npy", allow_pickle=True).item()

    for i, command in commands.items():
        i = int(i)
        dt, sc, ac, vel = command
        dx, dy, dyaw = motion_model([sc * 0.3, ac, vel], dt)

        persistent_map = rotate_translate_image(persistent_map, -dyaw * 180 / np.pi, dx, dy)

        front = cv2.imread(f"{dataset_location}/masks/{i}_front.png")
        left = cv2.imread(f"{dataset_location}/masks/{i}_left.png")
        right = cv2.imread(f"{dataset_location}/masks/{i}_right.png")
        if i in track_maps.keys():
            track_map = track_maps[i]
            if (track_map["left"] is not None) and (track_map["right"] is not None):
                map_bev = track_map_to_bev(track_map)
            else:
                map_bev = None

        else:
            map_bev = None

        front = np.where(front == (109, 80, 204), 1, 0).astype(np.uint8)
        front = front[:, :, 1]
        left = np.where(left == (109, 80, 204), 1, 0).astype(np.uint8)
        left = left[:, :, 1]
        right = np.where(right == (109, 80, 204), 1, 0).astype(np.uint8)
        right = right[:, :, 1]

        # cv2.imshow("front_original", front * 255)
        # cv2.imshow("left_original", left * 255)
        # cv2.imshow("right_original", right * 255)

        bev_size = (size, size)
        front = cv2.warpPerspective(front, H_front, bev_size)
        left = cv2.warpPerspective(left, H_left, bev_size)
        right = cv2.warpPerspective(right, H_right, bev_size)
        imgs = np.stack([front, left, right])

        # cv2.imshow("front", front * 255)
        # cv2.imshow("left", left * 255)
        # cv2.imshow("right", right * 255)

        image_certainty = np.flip(np.exp(-np.arange(0, front.shape[0]) / 25))
        localisation_certainty = np.flip(np.exp(-np.arange(0, front.shape[0]) / 50))

        image_certainty = np.arange(0, front.shape[0]) / front.shape[0]
        localisation_certainty = np.arange(0, front.shape[0]) / front.shape[0]

        imgs = np.multiply(imgs.transpose(0, 2, 1), image_certainty).transpose(0, 2, 1)
        image_map = np.max(imgs, axis=0)

        if map_bev is not None:
            localisation_map = np.multiply(map_bev.transpose(1, 0), localisation_certainty).transpose(1, 0)
        # map_certainty = map_certainty / np.max(map_certainty)

        # persistent_map = np.add(persistent_map * 0.5, map_certainty * 0.2)
        # persistent_map = np.add(persistent_map * 1.0, map_bev * 0.2)

        persistent_map = persistent_map * 0.5
        ratio = 1.0

        if i % 1 == 0:
            if map_bev is not None:
                localisation_map = np.mean(np.stack([image_map, localisation_map]), axis=0)
                persistent_map = np.sum(np.stack([persistent_map * 1.0, localisation_map * ratio]), axis=0)

        persistent_map = np.sum(np.stack([persistent_map * 1.0, image_map * ratio]), axis=0)

        # persistent_map /= np.max(persistent_map)

        cv2.imshow("Map", image_map * 255 / np.max(image_map))
        cv2.imshow("Persistent map", np.where(persistent_map > 0.9, 1, 0).astype(np.uint8) * 255)
        # cv2.imshow("Persistent map", persistent_map * 255)
        cv2.waitKey(50)


if __name__ == "__main__":
    image_mapping()
