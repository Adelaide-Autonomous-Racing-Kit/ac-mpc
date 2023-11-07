import os
from typing import Dict

from perception.utils import smooth_track_with_polyfit
from utils import load


class TrackLimits:
    def __init__(self, data_path: str):
        self._data_path = data_path

    def __getitem__(self, index: int) -> Dict:
        track = self._load_track_limits_detection(index)
        number_of_points = 300
        for track_name, points in track.items():
            track[track_name] = smooth_track_with_polyfit(points, number_of_points)
        return track

    def _load_track_limits_detection(self, index: int) -> Dict:
        filepath = os.path.join(self._data_path, "maps", f"{index}.npy")
        return load.npy(filepath)
