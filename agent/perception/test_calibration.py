import copy


import numpy as np
import cv2

from utils import CameraInfo, DUMMY_GROUND_POINTS

CAMERA_CONFIG = {
    "image_width": 1080,
    "image_height": 540,
    "position": [0.000102, 2.099975, 0.7],
    # "position": [200, 400, 0.7],
    "rotation_deg": [90.0, 0.0, 0.0],
    "vertical_fov_deg":60
}

BEV_ROAD = np.zeros([400,400, 3], np.uint8)
BEV_ROAD[:, 150:200] = (255,0,0)
BEV_ROAD[:, 200:250] = (0, 0, 255)

BEV_POINTS = np.array([
    [150, 0, 1],
    [250, 0, 1],
    [150, 400, 1],
    [200, 400, 1]
])

GROUND_POINTS = np.array([
    [50, 400, 1],
    [-50, 400, 1],
    [50, 0, 1],
    [0, 0, 1]
])

thickness = 2
IMAGE_OF_ROAD = np.zeros([CAMERA_CONFIG["image_height"], CAMERA_CONFIG["image_width"], 3], np.uint8)

# cv2.line(IMAGE_OF_ROAD, [0,CAMERA_CONFIG["image_height"]], [CAMERA_CONFIG["image_width"]//2-5, CAMERA_CONFIG["image_height"]//2], (255, 0, 0), thickness)
# cv2.line(IMAGE_OF_ROAD, [CAMERA_CONFIG["image_width"], CAMERA_CONFIG["image_height"]], [CAMERA_CONFIG["image_width"]//2+5, CAMERA_CONFIG["image_height"]//2], (0, 0, 255), thickness)

track_points = 100
start = 0
finish = 100
width = 10
left_track_wcf = np.array([np.ones(track_points) * -width/2, np.linspace(start, finish, track_points), np.ones(track_points)])
right_track_wcf = np.array([np.ones(track_points) * width/2, np.linspace(start, finish, track_points), np.ones(track_points)])

IMAGE_POINTS = []

# def test_homography_consistency()

# def test_overhead_camera_reprojection_calculations()

def test_homography_projection():

    camera_config = copy.deepcopy(CAMERA_CONFIG)
    camera_info = CameraInfo(camera_config)

    homography_w2i = np.linalg.inv(camera_info.homography_i2w)
    left_track_image = transform_points_wcf_to_image(homography_w2i, left_track_wcf)
    right_track_image = transform_points_wcf_to_image(homography_w2i, right_track_wcf)

    draw_tracks_on_image("original", left_track_image, right_track_image)

    camera_config = copy.deepcopy(CAMERA_CONFIG)
    rotation_amount = -20
    camera_config["rotation_deg"][2] += rotation_amount
    camera_info = CameraInfo(camera_config)

    homography_w2i = np.linalg.inv(camera_info.homography_i2w)
    left_track_image = transform_points_wcf_to_image(homography_w2i, left_track_wcf)
    right_track_image = transform_points_wcf_to_image(homography_w2i, right_track_wcf)

    draw_tracks_on_image(f"pan rotation by {rotation_amount} deg", left_track_image, right_track_image)

    camera_config = copy.deepcopy(CAMERA_CONFIG)
    rotation_amount = 20
    camera_config["rotation_deg"][2] += rotation_amount
    camera_info = CameraInfo(camera_config)

    homography_w2i = np.linalg.inv(camera_info.homography_i2w)
    left_track_image = transform_points_wcf_to_image(homography_w2i, left_track_wcf)
    right_track_image = transform_points_wcf_to_image(homography_w2i, right_track_wcf)

    draw_tracks_on_image(f"pan rotation by {rotation_amount} deg", left_track_image, right_track_image)

    camera_config = copy.deepcopy(CAMERA_CONFIG)
    rotation_amount = -10
    camera_config["rotation_deg"][1] += rotation_amount
    camera_info = CameraInfo(camera_config)

    homography_w2i = np.linalg.inv(camera_info.homography_i2w)
    left_track_image = transform_points_wcf_to_image(homography_w2i, left_track_wcf)
    right_track_image = transform_points_wcf_to_image(homography_w2i, right_track_wcf)

    draw_tracks_on_image(f"roll rotation by {rotation_amount} deg", left_track_image, right_track_image)

    camera_config = copy.deepcopy(CAMERA_CONFIG)
    rotation_amount = 10
    camera_config["rotation_deg"][1] += rotation_amount
    camera_info = CameraInfo(camera_config)

    homography_w2i = np.linalg.inv(camera_info.homography_i2w)
    left_track_image = transform_points_wcf_to_image(homography_w2i, left_track_wcf)
    right_track_image = transform_points_wcf_to_image(homography_w2i, right_track_wcf)

    draw_tracks_on_image(f"roll rotation by {rotation_amount} deg", left_track_image, right_track_image)

    camera_config = copy.deepcopy(CAMERA_CONFIG)
    rotation_amount = 10
    camera_config["rotation_deg"][0] += rotation_amount
    camera_info = CameraInfo(camera_config)

    homography_w2i = np.linalg.inv(camera_info.homography_i2w)
    left_track_image = transform_points_wcf_to_image(homography_w2i, left_track_wcf)
    right_track_image = transform_points_wcf_to_image(homography_w2i, right_track_wcf)

    draw_tracks_on_image(f"tilt rotation by {rotation_amount} deg", left_track_image, right_track_image)

    camera_config = copy.deepcopy(CAMERA_CONFIG)
    rotation_amount = -10
    camera_config["rotation_deg"][0] += rotation_amount
    camera_info = CameraInfo(camera_config)

    homography_w2i = np.linalg.inv(camera_info.homography_i2w)
    left_track_image = transform_points_wcf_to_image(homography_w2i, left_track_wcf)
    right_track_image = transform_points_wcf_to_image(homography_w2i, right_track_wcf)

    draw_tracks_on_image(f"tilt rotation by {rotation_amount} deg", left_track_image, right_track_image)

    cv2.waitKey(0)


def transform_points_wcf_to_image(homography_w2i, wcf_points):
    image_points = homography_w2i @ wcf_points
    image_points = image_points[:2] / image_points[2]

    x_mask = (image_points[0] > 0) & (image_points[0] < CAMERA_CONFIG["image_width"])
    y_mask = (image_points[1] > 0) & (image_points[1] < CAMERA_CONFIG["image_height"])
    mask = x_mask & y_mask

    filtered_image_points = image_points[:, mask]

    print(filtered_image_points)

    return filtered_image_points

def draw_tracks_on_image(window_name, left_track, right_track):
    image_of_road = copy.copy(IMAGE_OF_ROAD)
    cv2.polylines(image_of_road, [left_track.astype(np.int32).T], False, (255, 0, 0), thickness)
    cv2.polylines(image_of_road, [right_track.astype(np.int32).T], False, (0,0,255), thickness)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image_of_road)



def test_default_rotation():

    print(BEV_POINTS - GROUND_POINTS)

    camera_config = copy.copy(CAMERA_CONFIG)
    # camera_config["rotation_deg"][0] = 0
    camera_info = CameraInfo(camera_config)

    homography_w2bev = cv2.findHomography(GROUND_POINTS, BEV_POINTS)[0]

    print(homography_w2bev)

    # homography_w2i = np.linalg.inv(camera_info.homography_i2w)

    warped_wcf = cv2.warpPerspective(IMAGE_OF_ROAD, camera_info.homography_i2w , dsize=(400,400))
    warped_BEV = cv2.warpPerspective(warped_wcf, homography_w2bev, dsize=(400,400))

    homography_im2bev = homography_w2bev * camera_info.homography_i2w
    single_transform = cv2.warpPerspective(IMAGE_OF_ROAD, homography_im2bev, dsize=(400,400))

    cv2.namedWindow("warped_wcf", cv2.WINDOW_NORMAL)
    cv2.namedWindow("warped_bev", cv2.WINDOW_NORMAL)
    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("single_transform", cv2.WINDOW_NORMAL)
    cv2.imshow("warped_wcf", warped_wcf)
    cv2.imshow("original", IMAGE_OF_ROAD)
    cv2.imshow("warped_bev", warped_BEV)
    cv2.imshow("single_transform", single_transform)
    cv2.waitKey(0)



# def test_pitch_rotation():



# def test_roll_rotation():



def test_image_to_ground_projection():
    camera_info = CameraInfo(CAMERA_CONFIG)
    image_points = camera_info._get_corresponding_image_point(DUMMY_GROUND_POINTS)

    projected_ground_points = np.matmul(camera_info.homography_i2w, image_points.T)
    projected_ground_points =projected_ground_points[:2] / projected_ground_points[2]
    projected_ground_points = projected_ground_points.T

    print(projected_ground_points)
    print(DUMMY_GROUND_POINTS)

    assert np.all(np.isclose(projected_ground_points, DUMMY_GROUND_POINTS[:, :2])), "Homography "


if __name__ == "__main__":
    # test_default_rotation()

    test_image_to_ground_projection()
    test_homography_projection()