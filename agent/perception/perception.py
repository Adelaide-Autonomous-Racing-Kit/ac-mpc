import io
from typing import Dict, List

import cv2
import numpy as np
from loguru import logger
from PIL import Image
from turbojpeg import TJPF_BGRX, TurboJPEG

TURBO_JPEG = TurboJPEG()

from perception.segmentation import TrackSegmenter
from perception.tracks import TrackLimitPerception
from perception.observations import ObservationDict


class Perceiver:
    def __init__(self, agent, cfg: Dict, test: bool = False):
        self.agent = agent
        self.cfg = cfg
        self.track_segmenter = TrackSegmenter(cfg)
        self.track_limit_exctractor = TrackLimitPerception(cfg, test)
        self.image_width = cfg["image_width"]
        self.image_height = cfg["image_height"]

    def perceive(self, obs: List):
        output = self.preprocess_observations(obs)
        self.add_track_limits(output)
        return output

    def preprocess_observations(self, obs: List) -> Dict:
        output_obs = ObservationDict(obs)
        self.assert_image_size_is_correct(output_obs)
        self.track_segmenter.add_inferred_segmentation_masks(output_obs)
        self.agent._maybe_draw_visualisations(output_obs)
        return output_obs

    def assert_image_size_is_correct(self, obs: Dict):
        obs["CameraFrontRGB"] = np.asarray(
            Image.open(
                io.BytesIO(
                    TURBO_JPEG.encode(obs["CameraFrontRGB"], pixel_format=TJPF_BGRX)
                )
            )
        )  # [..., ::-1]
        if not self.is_image_size_correct(obs["CameraFrontRGB"]):
            obs["CameraFrontRGB"] = cv2.resize(
                obs["CameraFrontRGB"],
                dsize=(self.image_width, self.image_height),
                interpolation=cv2.INTER_CUBIC,
            )

    def is_image_size_correct(self, image: np.array) -> bool:
        return image.shape[:2] == (
            self.image_height,
            self.image_width,
        )

    def add_track_limits(self, obs: Dict):
        output = self.track_limit_exctractor.extract_track_from_observations(obs)
        obs["tracks"] = output
