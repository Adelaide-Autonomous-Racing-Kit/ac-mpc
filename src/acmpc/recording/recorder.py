import json
import os
from typing import Dict

import cv2
from loguru import logger
import numpy as np


class DataRecorder:
    def __init__(self, save_path, cfg: Dict):
        self.cfg = cfg
        self.image_count = 0
        self.n_image_samples = self.cfg["collect_images"]
        self.enable_collect_images = self.n_image_samples > 0
        self.save_path = save_path
        self.maybe_setup_directories()

    def maybe_setup_directories(self):
        if not self.enable_collect_images:
            return
        self.image_save_path = self.save_path + "/datacollection/images"
        self.mask_save_path = self.save_path + "/datacollection/masks"
        self.map_save_path = self.save_path + "/datacollection/maps"
        self.command_save_path = self.save_path + "/datacollection/commands"
        self.command_samples = {}
        paths = [
            self.image_save_path,
            self.mask_save_path,
            self.command_save_path,
            self.map_save_path,
        ]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    def maybe_record_data(
        self, obs: Dict, dt: float, steering_angle: float, acceleration: float
    ):
        if not self.enable_collect_images:
            return

        # Save results or collect data
        filename = self.image_count

        for key, image in obs.items():
            if "RGB" in key or "Segm" in key:
                self.save_image(key + f"_{self.image_count}", image)

        self.command_samples[filename] = {
            "dt": dt,
            "steering_angle": steering_angle,
            "acceleration": acceleration,
            "velocity": obs["full_pose"]["velocity"],
            "full_pose": obs["full_pose"],
        }

        track_limit_detections = {
            "centre": obs["tracks"]["centre"],
            "left": obs["tracks"]["left"],
            "right": obs["tracks"]["right"],
        }

        np.save(f"{self.map_save_path}/{filename}.npy", track_limit_detections)

        with open(f"{self.command_save_path}/commands.json", "w+") as command_file:
            command_file.write(json.dumps(self.command_samples))

        self.image_count += 1

        if self.image_count == self.n_image_samples - 1:
            self.enable_collect_images = False
            logger.info("Image collection completed")

    def save_image(self, name, image):
        if "RGB" in name:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{self.image_save_path}/{name}.png", image)
        else:
            cv2.imwrite(f"{self.mask_save_path}/{name}.png", image)
