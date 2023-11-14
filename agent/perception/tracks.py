import copy
from typing import Dict

import cv2
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

from perception import utils
from utils.track_limit_interpolation import maybe_interpolate_track_limit
from monitor.system_monitor import track_runtime

N_TRACK_POINTS_TO_RESAMPLE_FROM_POLY_FIT = 100


class TrackLimitPerception:
    def __init__(self, cfg: Dict, test: bool = False):
        self.cfg = cfg
        self.test = test
        self.use_interpolated_centreline = cfg["centerline_from_tack_limits"]
        self.image_width = cfg["image_width"]
        self.remove_bottom = 600
        self.birds_eye_view_dimension = 200  # m each length to form a square
        self.bev_scale = 4
        self.camera = utils.CameraInfo(cfg)

    @track_runtime
    def extract_track_from_observations(self, obs: Dict):
        tracks = self.process_image_masks(obs)
        # original_centre_track = copy.copy(tracks["centre"])
        tracks = self.process_track_points(tracks)
        return tracks

    # TODO: Change global variable to be part of object cfg
    def process_track_points(
        self,
        tracks: Dict,
        interpolate=True,
    ):
        num_points = N_TRACK_POINTS_TO_RESAMPLE_FROM_POLY_FIT
        # TODO: Make dict unpacking helper
        left_track = tracks["left"]
        right_track = tracks["right"]

        # Filters points that are in the fringes of the field of view
        mask = (
            (-50 < left_track[:, 0])
            & (left_track[:, 0] < 50)
            & (0 < left_track[:, 1])
            & (left_track[:, 1] < 150)
        )
        left_track = left_track[mask, :]
        mask = (
            (-50 < right_track[:, 0])
            & (right_track[:, 0] < 50)
            & (0 < right_track[:, 1])
            & (right_track[:, 1] < 150)
        )
        right_track = right_track[mask, :]

        if self.use_interpolated_centreline:
            track_limits = [left_track.T, right_track.T]
            maybe_interpolate_track_limit(track_limits)
            left_track, right_track = track_limits[0].T, track_limits[1].T
            left_track = utils.smooth_track_with_polyfit(left_track, num_points)
            right_track = utils.smooth_track_with_polyfit(right_track, num_points)
            centre_track = (right_track + left_track) / 2

        elif interpolate:
            # centre_track = np.concatenate([np.zeros((30, 2)), centre_track], axis=0)
            # centre_track = utils.smooth_track_with_polyfit(centre_track, num_points)
            left_track = utils.smooth_track_with_polyfit(left_track, num_points)
            right_track = utils.smooth_track_with_polyfit(right_track, num_points)
            centre_track = (right_track + left_track) / 2

        tracks = {
            "centre": centre_track,
            "left": left_track,
            "right": right_track,
        }
        return tracks

    def transform_track_image_points(self, columns):
        # remove track limit idxs that touch side of image or include bonnet of vehicle
        image_points = np.array(
            [
                [columns[row], row]
                for row in range(len(columns))
                if (columns[row] != 0)
                and (columns[row] != self.image_width - 1)
                and row < self.remove_bottom
            ]
        )
        if len(image_points) == 0:
            return np.zeros((0, 2))
        return self.camera.translate_points_from_image_to_ground_plane(image_points)

    def process_image_masks(self, observations):
        ascending_array = np.arange(1, observations["CameraFrontSegm"].shape[1] + 1)

        # Create empty track points so we can append any detected track points from any camera
        # We will sort them later into ascending y
        right_track_ground = np.zeros((0, 2))
        left_track_ground = np.zeros((0, 2))
        centre_track_ground = np.zeros((0, 2))

        if "CameraFrontSegm" in observations.keys():
            seg_mask = observations["CameraFrontSegm"]
            mask = np.multiply(seg_mask, ascending_array)
            right_columns = np.argmax(mask, axis=1)

            mask[mask == 0] = mask.shape[1] + 1
            left_columns = np.argmin(mask, axis=1)
            # centre_columns = (right_columns + left_columns) / 2

            if self.test:
                cv2.imshow("seg_centre", seg_mask * 255)

            right_track_ground = np.concatenate(
                [right_track_ground, self.transform_track_image_points(right_columns)],
                axis=0,
            )
            left_track_ground = np.concatenate(
                [left_track_ground, self.transform_track_image_points(left_columns)],
                axis=0,
            )
            # centre_track_ground = np.concatenate(
            #    [
            #        centre_track_ground,
            #        self.transform_track_image_points(centre_columns),
            #    ],
            #    axis=0,
            # )

        if "CameraLeftSegm" in observations.keys():
            seg_mask = observations["CameraLeftSegm"]
            mask = np.multiply(seg_mask, ascending_array)
            mask[mask == 0] = mask.shape[1] + 1
            left_columns = np.argmin(mask, axis=1)

            if self.test:
                cv2.imshow("seg_left", seg_mask * 255)

            left_track_ground = np.concatenate(
                [left_track_ground, self.transform_track_image_points(left_columns)],
                axis=0,
            )

        if "CameraRightSegm" in observations.keys():
            seg_mask = observations["CameraRightSegm"]
            mask = np.multiply(seg_mask, ascending_array)
            right_columns = np.argmax(mask, axis=1)

            if self.test:
                cv2.imshow("seg_right", seg_mask * 255)

            right_track_ground = np.concatenate(
                right_track_ground,
                [self.transform_track_image_points(right_columns)],
                axis=0,
            )

        # This sorting is needed if we are concatenating points from multiple cameras
        # Sorts into ascending y to then be smoothed out
        left_track_ground = left_track_ground[np.argsort(left_track_ground[:, 1])]
        right_track_ground = right_track_ground[np.argsort(right_track_ground[:, 1])]
        # centre_track_ground = centre_track_ground[np.argsort(centre_track_ground[:, 1])]

        if self.test:
            cv2.waitKey(0)
            fig, ax = plt.subplots()
            ax.scatter(right_track_ground[0], right_track_ground[1], label="right")
            ax.scatter(left_track_ground[0], left_track_ground[1], label="left")
            ax.scatter(centre_track_ground[0], centre_track_ground[1], label="centre")
            ax.arrow(0, 0, 0, 4, width=0.01)
            ax.legend()
            ax.set_aspect(1)
            plt.show()
        tracks = {
            # "centre": centre_track_ground,
            "left": left_track_ground,
            "right": right_track_ground,
        }
        return tracks
