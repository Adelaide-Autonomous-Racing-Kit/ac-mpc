from typing import Dict, Tuple

from concorde.tsp import TSPSolver
from loguru import logger
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist


class MapMaker:
    """Builds a map from segmentation left, centre, right tracks and the known x, y, yaw
    from the pose provided"""

    def __init__(self, verbose=False) -> None:
        self.map = {}
        self.xy_points_driven = []
        self.outside_track = []
        self.inside_track = []
        self.centre_track = []
        self.verbose = verbose
        self.map_built = False

    def map_world_pose_xy(self, processed_pose):
        xy_positions = [-processed_pose["x"], processed_pose["z"]]
        self.xy_points_driven.append(xy_positions)
        return xy_positions

    @staticmethod
    def transform_track_point(point, translation, rotation):
        transformed_point = np.matmul(rotation.T, point.T).T
        point = translation + transformed_point
        return point

    def process_segmentation_tracks(
        self, full_pose, left_track, right_track, centre_track
    ):
        """Convert ego pose to world pose for mapping of the width of the track"""
        # Get points that correspond to the minimal y value, to determine
        # track width at current point
        translation = self.map_world_pose_xy(full_pose)
        yaw = full_pose["translation_yaw"]
        map_rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

        leftmost_point = self.transform_track_point(left_track, translation, map_rot)
        centre_track = self.transform_track_point(centre_track, translation, map_rot)
        rightmost_point = self.transform_track_point(right_track, translation, map_rot)
        self.outside_track.append(leftmost_point)
        self.centre_track.append(centre_track)
        self.inside_track.append(rightmost_point)

    @staticmethod
    def earlier_points_come_before_later(
        ordered_points: np.array,
        unordered_points: np.array,
    ) -> np.array:
        """Checks to see that the ordered list has points that do come later in the original scan"""
        soon_idx = 0
        later_idx = 5
        ref_idx = 10
        sooner_point_check = np.linalg.norm(
            ordered_points[soon_idx] - unordered_points[ref_idx]
        )
        later_point_check = np.linalg.norm(
            ordered_points[later_idx] - unordered_points[ref_idx]
        )
        outside_points_should_go_forward = (
            sooner_point_check.mean() > later_point_check.mean()
        )

        return outside_points_should_go_forward

    def _flatten_array(self, array: np.array) -> np.array:
        flat = []
        for points_at_timestep in array:
            flat.extend(points_at_timestep[:1, :])
        return np.array(flat)

    def _save_raw_points(self, filename: str, insides: np.array, outsides: np.array):
        raw_points = {}
        points_filename = (
            f"{filename.split('.')[0]}-raw-points.{filename.split('.')[-1]}"
        )
        raw_points["outsides"] = np.copy(outsides)
        raw_points["insides"] = np.copy(insides)
        np.save(points_filename, raw_points, allow_pickle=True)

    def _calculate_center_track(
        self, outsides: np.array, insides: np.array
    ) -> np.array:

        distances = cdist(insides, outsides)
        centres = MapMaker.order_points(
            (insides + outsides[np.argmin(distances, axis=1)]) / 2
        )
        return centres

    def _smooth_track(self, track: np.array) -> np.array:
        smooth_x = MapMaker.smooth_boi(track, 0)
        smooth_y = MapMaker.smooth_boi(track, 1)
        return np.stack((smooth_x, smooth_y), axis=1)

    def _maybe_flip_outsides(self, outsides: np.array) -> np.array:
        return self._maybe_flip_track(outsides, self.outside_track)

    def _maybe_flip_insides(self, insides: np.array) -> np.array:
        return self._maybe_flip_track(insides, self.inside_track)

    def _maybe_flip_centres(self, centres: np.array) -> np.array:
        return self._maybe_flip_track(centres, self.inside_track)

    def _maybe_flip_track(self, track: np.array, raw_points: np.array) -> np.array:
        if not self.earlier_points_come_before_later(track, raw_points):
            track = np.flip(track, axis=0)
        return track

    def _remove_duplicate_points(
        self,
        centres: np.array,
        insides: np.array,
        outsides: np.array,
    ) -> Tuple[np.array]:
        # Remove near duplicate centre points
        d = np.diff(centres, axis=0)
        dists = np.hypot(d[:, 0], d[:, 1])
        is_not_duplicated = np.ones(dists.shape[0] + 1).astype(bool)
        is_not_duplicated[1:] = dists > 0.0001
        outsides = outsides[is_not_duplicated]
        insides = insides[is_not_duplicated]
        centres = centres[is_not_duplicated]
        return centres, insides, outsides

    def build_map(self, insides: np.array, outsides: np.array) -> Dict:
        logger.info("Processing outside Track Points")
        outsides = MapMaker.order_points(outsides)
        logger.info("Processing inside Track Points")
        insides = MapMaker.order_points(insides)
        logger.info("Processing center Track Points")
        centres = self._calculate_center_track(outsides, insides)

        outsides = self._smooth_track(outsides)
        centres = self._smooth_track(centres)
        insides = self._smooth_track(insides)

        outsides, centres, insides = map(
            MapMaker.order_points, (outsides, centres, insides)
        )

        self._maybe_flip_outsides(outsides)
        self._maybe_flip_insides(insides)
        self._maybe_flip_centres(centres)

        outsides = self.upsample_track(outsides)
        insides = self.upsample_track(insides)
        centres = self.upsample_track(centres)

        centres, insides, outsides = self._remove_duplicate_points(
            centres,
            insides,
            outsides,
        )

        output_map = {
            "outside_track": outsides,
            "inside_track": insides,
            "centre_track": centres,
        }
        return output_map

    def save_map(self, filename):
        # Slices remove starting and finishing points where vehicle is stationary
        outsides = self._flatten_array(self.outside_track)[30:-30]
        insides = self._flatten_array(self.inside_track)[30:-30]
        self._save_raw_points(filename, insides, outsides)
        output_map = self.build_map(insides, outsides)

        np.save(filename, output_map, allow_pickle=True)
        self.map_built = True

    @staticmethod
    def order_points(points: np.array) -> np.array:
        solver = TSPSolver.from_data(
            xs=points[:, 0],
            ys=points[:, 1],
            norm="EUC_2D",
        )

        soln = solver.solve(verbose=False, random_seed=42, time_bound=5)
        path = np.array(soln.tour, dtype=int)
        ordered_points = points[path, :]

        return ordered_points

    @staticmethod
    def smooth_boi(arr, i, window_length=15, polyorder=1):
        return savgol_filter(
            [p[i] for p in arr],
            window_length=window_length,
            polyorder=polyorder,
            mode="wrap",
        )

    @staticmethod
    def upsample_track(track, desired_density=0.5):

        distances = np.linalg.norm(track[1:] - track[:-1], axis=1)
        distance_between_map_points = np.mean(distances)
        upsample = np.ceil(distance_between_map_points / desired_density).astype(
            np.int32
        )

        output_centre = None

        for i in range(len(track) - 1):
            current_point = track[i]
            next_point = track[i + 1]
            new_xs = np.linspace(current_point[0], next_point[0], upsample)[:-1]
            new_ys = np.linspace(current_point[1], next_point[1], upsample)[:-1]
            new_points = np.array([new_xs, new_ys]).T

            if output_centre is not None:
                output_centre = np.concatenate([output_centre, new_points])
            else:
                output_centre = new_points

        return output_centre
