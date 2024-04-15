import multiprocessing as mp
import io
import signal
from typing import Dict

import cv2
import numpy as np
from loguru import logger
from PIL import Image
from turbojpeg import TJPF_BGRX, TurboJPEG


from perception.observations import ObservationDict
from perception.tracks import TrackLimitPerception
from perception.segmentation import TrackSegmenter
from perception.shared_memory import SharedImage, SharedPoints

TURBO_JPEG = TurboJPEG()


class Perceiver:
    def __init__(self, cfg: Dict):
        self._perceiver = PerceptionProcess(cfg)
        self.image_width = cfg["image_width"]
        self.image_height = cfg["image_height"]
        self._perceiver.start()
        self._is_first_resize = True

    @property
    def is_centreline_stale(self) -> bool:
        return self._perceiver.is_centreline_stale

    @property
    def centreline(self) -> np.array:
        return self._perceiver.centreline

    @property
    def is_tracklimits_stale(self) -> bool:
        return self._perceiver.is_tracklimits_stale

    @property
    def tracklimits(self) -> Dict:
        return self._perceiver.tracklimits

    @property
    def visualisation_tracks(self) -> Dict:
        return self._perceiver.tracks

    @property
    def input_image(self) -> np.array:
        return self._perceiver.input_image

    @property
    def output_mask(self) -> np.array:
        return self._perceiver.output_mask

    @property
    def output_visualisation(self) -> np.array:
        return self._perceiver.output_visualisation

    def perceive(self, obs: ObservationDict):
        self._preprocess_observations(obs)
        self._submit_image_to_perceiver(obs)

    def _preprocess_observations(self, obs: ObservationDict):
        obs["CameraFrontRGB"] = self._encode_decode_image(obs["CameraFrontRGB"])
        self._assert_image_size_is_correct(obs)

    def _encode_decode_image(self, image: np.array) -> np.array:
        # Model was trained on jpgs, this keeps it consistent
        image = np.asarray(
            Image.open(
                io.BytesIO(
                    TURBO_JPEG.encode(
                        image,
                        pixel_format=TJPF_BGRX,
                    )
                )
            )
        )
        return image

    def _assert_image_size_is_correct(self, obs: ObservationDict):
        if not self._is_image_size_correct(obs["CameraFrontRGB"]):
            original_size = obs["CameraFrontRGB"].shape
            if self._is_first_resize:
                message = "Resizing images from  "
                message = f"{original_size[0]}x{original_size[1]} "
                message += f"to {self.image_width}x{self.image_height}."
                logger.warning(message)
                self._is_first_resize = False
            obs["CameraFrontRGB"] = cv2.resize(
                obs["CameraFrontRGB"],
                dsize=(self.image_width, self.image_height),
                interpolation=cv2.INTER_CUBIC,
            )

    def _is_image_size_correct(self, image: np.array) -> bool:
        return image.shape[:2] == (self.image_height, self.image_width)

    def _submit_image_to_perceiver(self, obs: ObservationDict):
        self._perceiver.input_image = obs["CameraFrontRGB"]

    def shutdown(self):
        self._perceiver.is_running = False


class PerceptionProcess(mp.Process):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.__setup(cfg)

    @property
    def is_running(self) -> bool:
        """
        Checks if the segmentation process is running

        :return: True if the process is running, false if it is not
        :rtype: bool
        """
        with self._is_running.get_lock():
            is_running = self._is_running.value
        return is_running

    @is_running.setter
    def is_running(self, is_running: bool):
        """
        Sets if the segmentation process is running

        :is_running: True if the process is running, false if it is not
        :type is_running: bool
        """
        with self._is_running.get_lock():
            self._is_running.value = is_running

    @property
    def is_tracklimits_stale(self) -> bool:
        """
        Checks if the current tracklimits have been read by any consumer

        :return: True if the tracklimits have been read, false if it has not
        :rtype: bool
        """
        with self._is_tracklimits_stale.get_lock():
            is_stale = self._is_tracklimits_stale.value
        return is_stale

    @property
    def is_centreline_stale(self) -> bool:
        """
        Checks if the current centreline has been read by any consumer

        :return: True if the centreline has been read, false if it has not
        :rtype: bool
        """
        with self._is_centreline_stale.get_lock():
            is_stale = self._is_centreline_stale.value
        return is_stale

    @property
    def input_image(self) -> np.array:
        return self._shared_input.image

    @input_image.setter
    def input_image(self, image: np.array):
        self._shared_input.image = image

    @property
    def output_mask(self) -> np.array:
        return self._shared_mask.image

    @property
    def output_visualisation(self) -> np.array:
        return self._shared_visualisation.image

    @property
    def is_mask_stale(self) -> bool:
        return self._shared_mask.is_stale

    @property
    def tracklimits(self) -> Dict:
        with self._is_tracklimits_stale.get_lock():
            tracks = {
                "left": self._shared_left_track.points,
                "right": self._shared_right_track.points,
            }
            self._is_tracklimits_stale.value = True
        return tracks

    @property
    def centreline(self) -> Dict:
        with self._is_centreline_stale.get_lock():
            centerline = self._shared_centre_track.points
            self._is_centreline_stale.value = True
        return centerline

    @property
    def tracks(self) -> Dict:
        tracks = {}
        with self._is_tracklimits_stale.get_lock():
            tracks["left"] = self._shared_left_track.points
            tracks["right"] = self._shared_right_track.points
        with self._is_centreline_stale.get_lock():
            tracks["centre"] = self._shared_centre_track.points
        return tracks

    def run(self):
        """
        Called on PerceptionProcess.start()
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self._segmenter._setup_segmentation_model()
        while self.is_running:
            mask = self._segment_drivable_area()
            self._extract_tracklimits(mask)

    def _segment_drivable_area(self) -> np.array:
        image = self._shared_input.fresh_image
        mask, vis = self._segmenter.segment_drivable_area(image)
        self._shared_mask.image = mask
        self._shared_visualisation.image = vis
        return mask

    def _extract_tracklimits(self, mask: np.array):
        tracks = self._tracklimit_extractor(mask)
        self._update_centreline(tracks)
        self._update_tracklimits(tracks)

    def _update_tracklimits(self, tracks: Dict):
        with self._is_tracklimits_stale.get_lock():
            self._shared_left_track.points = tracks["left"]
            self._shared_right_track.points = tracks["right"]
            self._is_tracklimits_stale.value = False

    def _update_centreline(self, tracks: Dict):
        with self._is_centreline_stale.get_lock():
            self._shared_centre_track.points = tracks["centre"]
            self._is_centreline_stale.value = False

    def __setup(self, cfg: Dict):
        self.__setup_config(cfg)
        self.__setup_segmenter(cfg)
        self.__setup_track_extractor(cfg)
        self.__setup_shared_memory()

    def __setup_config(self, cfg: Dict):
        self._width = cfg["image_width"]
        self._height = cfg["image_height"]
        self._n_polyfit_points = cfg["n_polyfit_points"]

    def __setup_segmenter(self, cfg: Dict):
        self._segmenter = TrackSegmenter(cfg)

    def __setup_track_extractor(self, cfg: Dict):
        self._tracklimit_extractor = TrackLimitPerception(cfg)

    def __setup_shared_memory(self):
        self._shared_input = SharedImage(self._height, self._width, 3)
        self._shared_mask = SharedImage(1, self._height, self._width)
        self._shared_visualisation = SharedImage(1, self._height, self._width)
        self._shared_left_track = SharedPoints(self._n_polyfit_points, 2)
        self._shared_right_track = SharedPoints(self._n_polyfit_points, 2)
        self._shared_centre_track = SharedPoints(self._n_polyfit_points, 2)
        self._is_running = mp.Value("i", True)
        self._is_tracklimits_stale = mp.Value("i", True)
        self._is_centreline_stale = mp.Value("i", True)
