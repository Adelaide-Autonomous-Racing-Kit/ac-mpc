import os
import time
from typing import Dict, List

from utils import load

from localisation.localisation import LocaliseOnTrack
from localisation.tracker import LocalisationTracker
from localisation.utils import TrackLimits
from localisation.visualisation import LocalisationVisualiser


class BenchmarkLocalisation:
    def __init__(self, cfg: Dict):
        self._cfg = cfg
        self._test_data = self._load_commands_data(cfg["data_path"])
        self._track_limit_detections = TrackLimits(cfg["data_path"])
        self._setup_particle_filter()
        self._setup_tracker(self._localiser)
        self._setup_visualiser(self._localiser, self._tracker)

    def _load_commands_data(self, data_path: str) -> Dict:
        filepath = os.path.join(data_path, "commands", "commands.json")
        return load.json(filepath)

    def _setup_particle_filter(self) -> LocaliseOnTrack:
        racetrack_limits = load.track_map(self._cfg["map_path"])
        localiser = LocaliseOnTrack(
            racetrack_limits["centre_track"],
            racetrack_limits["outside_track"],
            racetrack_limits["inside_track"],
            self._cfg["particle_filter"],
        )
        self._localiser = localiser

    def _setup_tracker(self, localiser: LocaliseOnTrack) -> LocalisationTracker:
        self._tracker = LocalisationTracker(localiser, self._gt_poses)

    def _setup_visualiser(
        self,
        localiser: LocaliseOnTrack,
        tracker: LocalisationTracker,
    ):
        self._visualiser = LocalisationVisualiser(localiser, tracker)

    @property
    def _gt_poses(self) -> List[Dict]:
        poses = []
        for key in self._test_data.keys():
            poses.append(self._test_data[key]["full_pose"])
        return poses

    def run(self):
        for i, control_inputs in self._input_data.items():
            track_detections = self._track_limit_detections[i]
            time_for_step = self._step_localiser(control_inputs, track_detections)
            self._tracker.update(time_for_step)
            self._visualiser.update(track_detections)

    @property
    def _input_data(self) -> Dict:
        control_inputs = {}
        for key in self._test_data.keys():
            control_inputs[int(key)] = {
                "dt": self._test_data[key]["dt"],
                "steering": self._test_data[key]["steering_angle"],
                "acceleration": self._test_data[key]["acceleration"],
                "velocity": self._test_data[key]["velocity"],
            }
        return control_inputs

    def _step_localiser(self, control: Dict, track_detections: Dict):
        start_time = time.time()
        self._localiser.step(
            control_command=(
                control["steering"],
                control["acceleration"],
                control["velocity"],
            ),
            dt=control["dt"],
            observation=[track_detections["left"], track_detections["right"]],
        )
        return time.time() - start_time
