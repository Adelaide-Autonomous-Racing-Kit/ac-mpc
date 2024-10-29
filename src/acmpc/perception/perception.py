import io
import multiprocessing as mp
from typing import Dict

from PIL import Image
from aci.utils.system_monitor import track_runtime
from acmpc.perception.observations import ObservationDict
from acmpc.perception.segmentation import (
    Segmentation_Monitor,
    TrackSegmenter,
    TrackSegmenterTensorRT,
)
from acmpc.perception.shared_memory import SharedImage, SharedPoints
from acmpc.perception.tracks import Track_Limits_Monitor, TrackLimitPerception
from acmpc.worker.base import WorkerProcess
import cv2
from loguru import logger
import numpy as np
from turbojpeg import TJPF_BGRX, TurboJPEG

TURBO_JPEG = TurboJPEG()


class Perceiver:
    def __init__(self, cfg: Dict):
        self.__setup(cfg)

    @property
    def is_centreline_stale(self) -> bool:
        return self._tracklimit_extractor.is_centreline_stale

    @property
    def centreline(self) -> np.array:
        return self._tracklimit_extractor.centreline

    @property
    def is_tracklimits_stale(self) -> bool:
        return self._tracklimit_extractor.is_tracklimits_stale

    @property
    def tracklimits(self) -> Dict:
        return self._tracklimit_extractor.tracklimits

    @property
    def visualisation_tracks(self) -> Dict:
        return self._tracklimit_extractor.tracks

    @property
    def input_image(self) -> np.array:
        return self._segmentor.input_image

    @property
    def output_mask(self) -> np.array:
        return self._segmentor.output_mask

    @property
    def output_visualisation(self) -> np.array:
        return self._segmentor.output_visualisation

    def perceive(self, obs: ObservationDict):
        if not obs["is_image_stale"]:
            self._preprocess_observations(obs)
            self._submit_image_to_segmentor(obs)

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
                message = f"{original_size[1]}x{original_size[0]} "
                message += f"to {self.image_width}x{self.image_height}."
                logger.warning(message)
                self._is_first_resize = False
            obs["CameraFrontRGB"] = cv2.resize(
                obs["CameraFrontRGB"],
                dsize=(self.image_width, self.image_height),
                interpolation=cv2.INTER_LINEAR,
            )

    def _is_image_size_correct(self, image: np.array) -> bool:
        return image.shape[:2] == (self.image_height, self.image_width)

    def _submit_image_to_segmentor(self, obs: ObservationDict):
        self._segmentor.input_image = obs["CameraFrontRGB"]

    def shutdown(self):
        self._segmentor.is_running = False
        self._tracklimit_extractor.is_running = False

    def __setup(self, cfg: Dict):
        self._setup_workers(cfg)
        self._unpack_config(cfg)
        self._launch_workers()
        self._is_first_resize = True

    def _setup_workers(self, cfg: Dict):
        self._segmentor = SegmentationProcess(cfg)
        self._tracklimit_extractor = TrackExtractionProcess(cfg, self._segmentor)

    def _unpack_config(self, cfg: Dict):
        self.image_width = cfg["image_width"]
        self.image_height = cfg["image_height"]

    def _launch_workers(self):
        self._segmentor.start()
        self._tracklimit_extractor.start()


class SegmentationProcess(WorkerProcess):
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

    def _runtime_setup(self):
        self._segmenter._setup_segmentation_model()

    def _work(self):
        frame = self._get_new_frame()
        self._segment_drivable_area(frame)
        # Segmentation_Monitor.maybe_log_function_itterations_per_second()

    def _get_new_frame(self) -> np.array:
        return self._shared_input.fresh_image

    def _segment_drivable_area(self, frame: np.array):
        mask, vis = self._segmenter.segment_drivable_area(frame)
        self._update_shared_images(mask, vis)

    def _update_shared_images(self, mask: np.array, vis: np.array):
        self._shared_mask.image = mask
        self._shared_visualisation.image = vis

    def _setup(self, cfg: Dict):
        self._unpack_config(cfg)
        self.__setup_segmenter(cfg)
        self._allocate_shared_memory()

    def _unpack_config(self, cfg: Dict):
        self._width = cfg["image_width"]
        self._height = cfg["image_height"]

    def __setup_segmenter(self, cfg: Dict):
        self._segmenter = TrackSegmenter(cfg)
        # self._segmenter = TrackSegmenterTensorRT(cfg)

    def _allocate_shared_memory(self):
        self._shared_input = SharedImage(self._height, self._width, 3)
        self._shared_mask = SharedImage(1, self._height, self._width)
        self._shared_visualisation = SharedImage(1, self._height, self._width)


class TrackExtractionProcess(WorkerProcess):
    def __init__(self, cfg: Dict, segmentor: SegmentationProcess):
        super().__init__(cfg)
        self._shared_mask = segmentor._shared_mask

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

    def _work(self):
        mask = self._get_new_mask()
        self._extract_tracklimits(mask)
        # Track_Limits_Monitor.maybe_log_function_itterations_per_second()

    def _get_new_mask(self) -> np.array:
        return np.squeeze(self._shared_mask.fresh_image, axis=0)

    @track_runtime(Track_Limits_Monitor)
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

    def _runtime_setup(self):
        pass

    def _setup(self, cfg: Dict):
        self._unpack_config(cfg)
        self._allocate_shared_memory()
        self.__setup_track_extractor(cfg)

    def _unpack_config(self, cfg: Dict):
        self._width = cfg["image_width"]
        self._height = cfg["image_height"]
        self._n_polyfit_points = cfg["n_polyfit_points"]

    def _allocate_shared_memory(self):
        self._shared_left_track = SharedPoints(self._n_polyfit_points, 2)
        self._shared_right_track = SharedPoints(self._n_polyfit_points, 2)
        self._shared_centre_track = SharedPoints(self._n_polyfit_points, 2)
        self._is_tracklimits_stale = mp.Value("i", True)
        self._is_centreline_stale = mp.Value("i", True)

    def __setup_track_extractor(self, cfg: Dict):
        self._tracklimit_extractor = TrackLimitPerception(cfg)
