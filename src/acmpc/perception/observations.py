from typing import Dict, List

import numpy as np
from acmpc.utils.radians import convert_radians_to_plus_minus_pi


class ObservationDict(dict):
    def __init__(self, obs: Dict, *arg, **kw):
        super().__init__(*arg, **kw)
        self._setup(obs)

    def get_images(self) -> List[np.array]:
        return [self["CameraFrontRGB"]]

    def add_segmentation_masks(self, masks: np.array):
        self["CameraFrontSegm"] = masks[0]

    def _setup(self, obs: Dict):
        self["is_image_stale"] = obs["is_image_stale"]
        self["CameraFrontRGB"] = obs["image"]
        pose = self._unpack_pose(obs["state"])
        self["speed"] = pose["velocity"]
        self["full_pose"] = pose
        self["i_current_time"] = obs["state"]["i_current_time"]
        self["i_best_time"] = obs["state"]["i_best_time"]
        self["i_last_time"] = obs["state"]["i_last_time"]
        self["current_sector_index"] = obs["state"]["current_sector_index"]
        self["completed_laps"] = obs["state"]["completed_laps"]
        self["last_sector_time"] = obs["state"]["last_sector_time"]

    def _unpack_pose(self, pose: Dict) -> Dict:
        pose = {
            "SteeringRequest": pose["steering_angle"],
            "GearRequest": pose["gear"].astype("float"),
            # "Mode": pose[2].astype("float"),
            "velocity": np.sqrt(
                pose["velocity_x"] ** 2
                + pose["velocity_y"] ** 2
                + pose["velocity_z"] ** 2
            ),
            "vx": pose["velocity_x"],
            "vy": pose["velocity_y"],
            "vz": pose["velocity_z"],
            "ax": pose["acceleration_g_X"],
            "ay": pose["acceleration_g_Y"],
            "az": pose["acceleration_g_Z"],
            "avx": pose["local_angular_velocity_X"],
            "avy": pose["local_angular_velocity_Y"],
            "avz": pose["local_angular_velocity_Z"],
            "yaw": convert_radians_to_plus_minus_pi(pose["heading"]),
            "pitch": pose["pitch"],
            "roll": pose["roll"],
            "x": pose["ego_location_x"],
            "y": pose["ego_location_y"],
            "z": pose["ego_location_z"],
            "translation_yaw": pose["heading"],
        }
        return pose
