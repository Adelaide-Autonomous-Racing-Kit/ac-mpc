from typing import Tuple

import cv2
import numpy as np

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
    ],
    dtype=np.uint8,
)


def draw_track_lines_on_bev(bev, scale, list_of_lines, colour=(255, 0, 255)):
    for track_line in list_of_lines:
        track_line = (track_line * scale).astype(np.int32)
        track_line[:, 0] = track_line[:, 0] + bev.shape[0] / 2
        draw_track_line(bev, track_line, colour, 2)


def draw_track_line(
    canvas: np.array,
    line: np.array,
    colour: Tuple[int],
    thickness: float,
):
    for i in range(len(line) - 2):
        cv2.line(
            canvas,
            line[i],
            line[i + 1],
            color=colour,
            thickness=thickness,
        )


def draw_arrow(
    canvas: np.array,
    start: np.array,
    direction: float,
    length: int,
    colour: Tuple[int],
    thickness: float,
):
    end = start + np.array(
        [
            length * np.cos(direction),
            length * np.sin(direction),
        ],
        dtype=np.int32,
    )
    cv2.arrowedLine(canvas, start, end, colour, thickness)


def transform_track_points(points, translation, rotation):
    points = points - translation
    return np.matmul(rotation, points.T).T


def get_track_points_by_index(indices: np.array, track: np.array) -> np.array:
    return track[np.mod(indices, len(track) - 1)]
