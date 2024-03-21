import cv2
import numpy as np
from loguru import logger


def draw_track_lines_on_bev(bev, scale, list_of_lines, colour=(255, 0, 255)):
    for track_line in list_of_lines:
        track_line = (track_line * scale).astype(np.int32)
        track_line[:, 0] = track_line[:, 0] + bev.shape[0] / 2
        track_line[:, 1] = bev.shape[1] / 2 + track_line[:, 1]

        for i in range(len(track_line) - 2):
            cv2.line(
                bev,
                track_line[i],
                track_line[i + 1],
                color=colour,
                thickness=2,
            )
            cv2.circle(bev, track_line[i], 2, colour, 2)


def transform_track_points(points, translation, rotation):
    points = points - translation
    return np.matmul(rotation, points.T).T


def get_track_points_by_index(indices: np.array, track: np.array) -> np.array:
    return track[np.mod(indices, len(track) - 1)]
