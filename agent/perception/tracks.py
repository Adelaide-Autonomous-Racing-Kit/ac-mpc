from typing import Dict

import numpy as np

from perception import utils
from utils.track_limit_interpolation import maybe_interpolate_track_limit
from monitor.system_monitor import track_runtime


class TrackLimitPerception:
    def __init__(self, cfg: Dict):
        self._use_interpolated_centreline = cfg["centerline_from_tack_limits"]
        self._image_width = cfg["image_width"]
        self._n_polyfit_points = cfg["n_polyfit_points"]
        self._n_rows_to_remove = cfg["n_rows_to_remove_bonnet"]
        self._camera = utils.CameraInfo(cfg)
    
    @track_runtime
    def __call__(self, mask: np.array) -> np.array:
        tracklimits = self._extract_tracklimits(mask)
        return self._process_track_points(tracklimits)
    
    def _extract_tracklimits(self, mask: np.array) -> Dict:
        ascending_array = np.arange(1, mask.shape[1] + 1, dtype=np.uint16)
        mask = np.multiply(mask, ascending_array)
        right_columns = np.argmax(mask, axis=1)
        mask[mask == 0] = mask.shape[1] + 1
        left_columns = np.argmin(mask, axis=1)

        right_track_ground = self._transform_track_image_points(right_columns)
        left_track_ground = self._transform_track_image_points(left_columns)

        tracks = {
            "left": left_track_ground,
            "right": right_track_ground,
        }
        return tracks
    
    def _transform_track_image_points(self, columns):
        # remove track limit idxs that touch side of image or include bonnet of vehicle
        image_points = np.array(
            [
                [columns[row], row]
                for row in range(len(columns))
                if (columns[row] != 0)
                and (columns[row] != self._image_width - 1)
                and row < self._n_rows_to_remove
            ]
        )
        if len(image_points) == 0:
            return np.zeros((0, 2))
        return self._camera.translate_points_from_image_to_ground_plane(image_points)

    def _process_track_points(self, tracks: Dict):
        num_points = self._n_polyfit_points
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

        if self._use_interpolated_centreline:
            track_limits = [left_track.T, right_track.T]
            maybe_interpolate_track_limit(track_limits)
            left_track, right_track = track_limits[0].T, track_limits[1].T
            left_track = utils.smooth_track_with_polyfit(left_track, num_points)
            right_track = utils.smooth_track_with_polyfit(right_track, num_points)
            centre_track = (right_track + left_track) / 2

        else:
            left_track = utils.smooth_track_with_polyfit(left_track, num_points)
            right_track = utils.smooth_track_with_polyfit(right_track, num_points)

            centre_track = (right_track + left_track) / 2
            origin_points = np.zeros((10, 2))
            origin_points[:,0] = centre_track[0][0]
            centre_track = np.concatenate([origin_points, centre_track], axis=0)
            centre_track = utils.smooth_track_with_polyfit(centre_track, num_points)

        tracks = {
            "centre": centre_track,
            "left": left_track,
            "right": right_track,
        }
        return tracks
