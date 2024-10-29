from collections import namedtuple
from typing import Dict, List, Tuple

from aci.utils.system_monitor import SystemMonitor, track_runtime
from acmpc.perception.utils import CameraInfo, smooth_track_with_polyfit
from acmpc.utils.track_limit_interpolation import maybe_interpolate_track_limit
import cv2
from loguru import logger
import numpy as np
from scipy.signal import savgol_filter

# TODO: Move into config or calculate
Limits = namedtuple("Limits", ["x_max", "x_min", "y_max", "y_min"])
BEV_FOV = Limits(x_max=50, x_min=-50, y_max=150, y_min=0)
MIN_CONTOUR_LENGTH = 500
WINDOW_SIZE = 30
NO_GAP_CONTOUR_Y_VALUE_INTERVAL = 25
CUT_SIZE = 2
MINIMUM_GAP_DISTANCE = 10

Track_Limits_Monitor = SystemMonitor(300)


class TrackLimitPerception:
    def __init__(self, cfg: Dict):
        self._use_interpolated_centreline = cfg["centerline_from_track_limits"]
        self._image_width = cfg["image_width"]
        self._image_height = cfg["image_height"]
        self._n_polyfit_points = cfg["n_polyfit_points"]
        self._n_rows_to_remove = cfg["n_rows_to_remove_bonnet"]
        self._camera = CameraInfo(cfg)

        self._min_contour_length = MIN_CONTOUR_LENGTH
        self._window_size = WINDOW_SIZE
        self._n_remove_for_contour_without_gap = NO_GAP_CONTOUR_Y_VALUE_INTERVAL
        self._n_trim = CUT_SIZE
        self._min_gap_distance = MINIMUM_GAP_DISTANCE
        self._split_methods = {
            0: self._split_contour_without_gap,
            1: self._split_contour_with_gap,
            2: self._split_contour_with_gaps,
        }

    def __call__(self, mask: np.array) -> np.array:
        tracklimits = self._extract_tracklimits(mask)
        return self._process_track_points(tracklimits)

    def _extract_tracklimits(self, mask: np.array) -> Dict:
        try:
            tracks = self._extract_tracklimits_from_contour(mask)
        except Exception as e:
            message = "Failed to extract from contour, falling back to naive extraction"
            logger.warning(f"{message} {e}")
            tracks = self._fallback_extract_tracklimits(mask)
        return tracks

    def _extract_tracklimits_from_contour(self, mask: np.array) -> Dict:
        contour = self._extract_contour(mask)
        left_track, right_track = self._split_contour(contour)
        left_track, right_track = self._post_process_tracks(left_track, right_track)
        tracks = {
            "left": self._transform_track_image_points(left_track),
            "right": self._transform_track_image_points(right_track),
        }
        return tracks

    def _extract_contour(self, mask: np.array) -> np.array:
        mask = mask[..., None] * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return self._select_largest_contour(contours)

    def _select_largest_contour(self, contours: List[np.array]) -> np.array:
        contours = self._preprocess_contours(contours)
        return sorted(contours, key=lambda x: contour_y_distance(x), reverse=True)[0]

    def _preprocess_contours(self, contours: List[np.array]) -> List[np.array]:
        contours = [
            contour for contour in contours if len(contour) > self._min_contour_length
        ]
        return [self._preprocess_contour(contour) for contour in contours]

    def _preprocess_contour(self, contour: np.array) -> np.array:
        contour = np.squeeze(contour, axis=1)
        contour = contour[
            (contour[:, 0] != 0)
            & (contour[:, 0] != self._image_width - 1)
            & (contour[:, 1] != 0)
            & (contour[:, 1] != self._image_height - 1)
            & (contour[:, 1] < self._n_rows_to_remove)
        ]
        return contour

    def _split_contour(self, contour: np.array) -> Tuple[np.array]:
        split_indices = self._find_gaps_in_contour(contour)
        n_gaps = len(split_indices)
        if n_gaps > 1:
            contour = extract_longest_segments(contour, split_indices)
            split_indices = self._find_gaps_in_contour(contour)[-2:]
            n_gaps = len(split_indices)
        return self._split_methods[n_gaps](contour, split_indices)

    def _find_gaps_in_contour(self, contour: np.array) -> np.array:
        inter_point_distances = np.linalg.norm(contour[1:] - contour[:-1], axis=1)
        is_gap = inter_point_distances > self._min_gap_distance
        split_indices = np.arange(0, contour.shape[0] - 1)[is_gap]
        split_indices.sort()
        return split_indices

    def _split_contour_with_gaps(
        self,
        contour: np.array,
        split_indices: np.array,
    ) -> Tuple[np.array]:
        component_1 = contour[: split_indices[0]]
        component_2 = contour[split_indices[0] + 1 : split_indices[1]]
        component_3 = contour[split_indices[1] + 1 :]
        left_track = join_line_segments(component_1, component_3)
        right_track = component_2
        return left_track, right_track

    def _split_contour_with_gap(
        self,
        contour: np.array,
        split_indices: np.array,
    ) -> Tuple[np.array]:
        component_1 = contour[0 : split_indices[0]]
        component_2 = contour[split_indices[0] + 1 :]
        contour = join_line_segments(component_1, component_2)
        inflection_index = self._find_distance_weighted_inflection_point(contour)
        right_track = contour[:inflection_index]
        left_track = contour[inflection_index + 1 :]
        return left_track, right_track

    def _find_distance_weighted_inflection_point(self, contour: np.array) -> np.array:
        dy_2 = savgol_filter(contour[:, 0], self._window_size, 2, 2)
        dx_2 = savgol_filter(contour[:, 1], self._window_size, 2, 2)
        derivative = np.abs(dy_2) + np.abs(dx_2)
        derivative /= contour[:, 1] - np.min(contour[:, 1]) + 1
        inflection_index = np.argmax(derivative)
        return inflection_index

    def _split_contour_without_gap(
        self,
        contour: np.array,
        split_indices: np.array,
    ) -> np.array:
        # Recursively remove low points until a large enough gap is created
        highest_y = np.max(contour[:, 1])
        y_threshold = highest_y - self._n_remove_for_contour_without_gap
        contour = contour[contour[:, 1] < y_threshold]
        return self._split_contour(contour)

    def _post_process_tracks(
        self,
        left_track: np.array,
        right_track: np.array,
    ) -> List[np.array]:
        right_track = self._filter_right_points(right_track)[: -self._n_trim]
        left_track = self._filter_left_points(left_track)[self._n_trim :]
        return left_track, right_track

    def _filter_right_points(self, right_track: np.array) -> np.array:
        # I should never see another x value lower on the same y value
        return self._remove_duplicate_y(right_track, is_right=True)

    def _filter_left_points(self, left_track: np.array) -> np.array:
        # I should never see another x value higher on the same y value
        return self._remove_duplicate_y(left_track, is_right=False)

    def _remove_duplicate_y(self, track: np.array, is_right: bool = True):
        _, indices = np.unique(track[:, 1], return_index=True)
        indices = np.sort(indices)
        res = np.split(track, indices)[1:]
        if is_right:
            track = [x[np.argmax(x[:, 0])] if len(x) > 1 else x[0] for x in res]
        else:
            track = [x[np.argmin(x[:, 0])] if len(x) > 1 else x[0] for x in res]
        return np.stack(track)

    def _fallback_extract_tracklimits(self, mask: np.array) -> Dict:
        ascending_array = np.arange(1, mask.shape[1] + 1, dtype=np.uint16)
        mask = np.multiply(mask, ascending_array)
        right_columns = np.argmax(mask, axis=1)
        mask[mask == 0] = mask.shape[1] + 1
        left_columns = np.argmin(mask, axis=1)

        left_track = self._get_image_points(left_columns)
        right_track = self._get_image_points(right_columns)

        tracks = {
            "left": self._transform_track_image_points(left_track),
            "right": self._transform_track_image_points(right_track),
        }
        return tracks

    @track_runtime(Track_Limits_Monitor)
    def _get_image_points(self, columns: np.array) -> np.array:
        # remove track limit idxs that touch side of image or include
        # the bonnet of vehicle
        points = [
            [column, i]
            for i, column in enumerate(columns)
            if (
                (column != 0)
                and (column != self._image_width - 1)
                and (i < self._n_rows_to_remove)
            )
        ]
        return np.array(points)

    def _transform_track_image_points(self, image_points: np.array) -> np.array:
        if len(image_points) == 0:
            return np.zeros((0, 2))
        return self._camera.translate_points_from_image_to_ground_plane(image_points)

    def _process_track_points(self, tracks: Dict):
        self._post_process_bev_tracks(tracks)
        if self._use_interpolated_centreline:
            maybe_interpolate_track_limit(tracks)
            tracks["left"] = self._smooth_track_with_polyfit(tracks["left"])
            tracks["right"] = self._smooth_track_with_polyfit(tracks["right"])
            tracks["centre"] = self._calculate_centre_track(tracks)
        else:
            tracks["left"] = self._smooth_track_with_polyfit(tracks["left"])
            tracks["right"] = self._smooth_track_with_polyfit(tracks["right"])
            tracks["centre"] = self._calculate_centre_track(tracks)

        return tracks

    def _post_process_bev_tracks(self, tracks: Dict):
        tracks["left"] = self._remove_points_outside_fov(tracks["left"])
        tracks["right"] = self._remove_points_outside_fov(tracks["right"])

    def _remove_points_outside_fov(self, track: np.array) -> np.array:
        # Filters points that are in the fringes of the field of view
        track = track[
            (BEV_FOV.x_min < track[:, 0])
            & (track[:, 0] < BEV_FOV.x_max)
            & (BEV_FOV.y_min < track[:, 1])
            & (track[:, 1] < BEV_FOV.y_max)
        ]
        return track

    def _smooth_track_with_polyfit(self, track: np.array) -> np.array:
        return smooth_track_with_polyfit(track, self._n_polyfit_points, 2)

    def _calculate_centre_track(self, tracks: Dict) -> np.array:
        centre_track = (tracks["left"] + tracks["right"]) / 2
        origin_points = np.zeros((10, 2))
        origin_points[:, 0] = centre_track[0][0]
        centre_track = np.concatenate([origin_points, centre_track], axis=0)
        return self._smooth_track_with_polyfit(centre_track)


@track_runtime(Track_Limits_Monitor)
def contour_bbox_area(contour: np.array) -> float:
    if contour.shape[0] == 0:
        return 0.0
    x_min, x_max = np.min(contour[:, 0]), np.max(contour[:, 0])
    y_min, y_max = np.min(contour[:, 1]), np.max(contour[:, 1])
    return (x_max - x_min) * (y_max - y_min)


def contour_y_distance(contour: np.array) -> float:
    if contour.shape[0] == 0:
        return 0.0
    y_min, y_max = np.min(contour[:, 1]), np.max(contour[:, 1])
    return abs(y_max - y_min)


def join_line_segments(segment_1: np.array, segment_2: np.array) -> np.array:
    distances = np.array(
        [
            np.linalg.norm(segment_1[-1] - segment_2[0]),
            np.linalg.norm(segment_1[0] - segment_2[-1]),
        ]
    )
    min_distance_index = np.argmin(np.abs(distances))
    if min_distance_index == 0:
        joined = np.concatenate([segment_1, segment_2])
    else:
        joined = np.concatenate([segment_2, segment_1])
    return joined


def extract_longest_segments(contour: np.array, split_indices: np.array) -> np.array:
    split_indices = np.append(split_indices, contour.shape[0] - 1)
    segment_y_distance = get_segment_y_distance(contour, split_indices)
    longest_indices = argmax_top_k(segment_y_distance, 2)
    return create_contour_from_longest_segments(contour, longest_indices, split_indices)


def get_segment_y_distance(contour: np.array, split_indices: np.array) -> np.array:
    start_index, y_distances = 0, []
    for split_index in split_indices:
        y_distance = np.abs(contour[start_index, 1] - contour[split_index, 1])
        y_distances.append(y_distance)
        start_index += split_index - start_index + 1
    return np.array(y_distances)


def argmax_top_k(array: np.array, k: int) -> np.array:
    return np.argpartition(array, array.shape[0] - k)[-k:]


def create_contour_from_longest_segments(
    contour: np.array,
    longest_indices: np.array,
    split_indices: np.array,
) -> np.array:
    new_contour, start_index = None, 0
    for i, split_index in enumerate(split_indices):
        if i in longest_indices:
            line_segment = contour[start_index:split_index]
            if new_contour is None:
                new_contour = line_segment
            else:
                new_contour = np.concatenate([new_contour, line_segment])
        start_index += split_index - start_index + 1
    return new_contour
