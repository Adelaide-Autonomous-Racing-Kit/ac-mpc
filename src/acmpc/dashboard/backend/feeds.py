from __future__ import annotations

import abc
from collections import namedtuple
import time
from typing import Dict

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from PyQt6.QtQuick import QQuickImageProvider
import cv2
from acmpc.dashboard.backend.utils import flip, flip_and_rotate
from acmpc.dashboard.visualisation.plots import (
    draw_control_map,
    draw_localisation_map,
    get_blank_canvas,
)
from acmpc.dashboard.visualisation.utils import COLOUR_LIST, draw_arrow, draw_track_line
import numpy as np
from acmpc.utils import load


class FeedThread(QThread):
    updateFrame = pyqtSignal(QImage)

    @abc.abstractmethod
    def _get_frame(self) -> np.array:
        """Define how the worker should produce a cv2 BGR image here"""
        pass

    @abc.abstractmethod
    def _pre_run_setup(self) -> np.array:
        """Define any setup before running main loop here"""
        pass

    @abc.abstractmethod
    def _teardown(self) -> np.array:
        """Define any clean up steps here"""
        pass

    def run(self):
        self.is_running = True
        self._pre_run_setup()
        while self.is_running:
            frame = self._get_frame()
            if frame is None:
                continue
            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(
                color_frame.data,
                color_frame.shape[1],
                color_frame.shape[0],
                QImage.Format.Format_RGB888,
            )
            self.updateFrame.emit(image)
            time.sleep(0.02)
        self._teardown()


class VideoThread(FeedThread):
    def __init__(self, source: str, parent=None):
        QThread.__init__(self, parent)
        self._source = source

    def _get_frame(self) -> np.array:
        ret, frame = self._capture.read()
        if not ret:
            return None
        return frame

    def _pre_run_setup(self) -> np.array:
        self._capture = cv2.VideoCapture(self._source)

    def _teardown(self):
        try:
            self._capture.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(e)


class VisualisationThread(FeedThread):
    def __init__(self, agent: ElTuarMPC, cfg: Dict, parent=None):
        QThread.__init__(self, parent)
        self._agent = agent
        self._unpack_config(cfg)

    def _unpack_config(self, cfg: Dict):
        self._cfg = cfg
        self._dimension = cfg["debugging"]["birds_eye_view_size"]
        self._scale = cfg["debugging"]["birds_eye_view_scale"]

    @abc.abstractmethod
    def _plot_visualisation(self) -> np.array:
        """Implement logic to create visualisation here"""
        pass

    def _get_frame(self) -> np.array:
        return self._plot_visualisation()

    def _pre_run_setup(self) -> np.array:
        pass

    def _teardown(self):
        pass


class CameraFeed(VisualisationThread):
    def _plot_visualisation(self) -> np.array:
        input_image = self._agent.perception.input_image
        return cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)


class SegmentationFeed(VisualisationThread):
    def _plot_visualisation(self) -> np.array:
        image = self._agent.perception.output_mask
        return np.transpose(image, axes=(1, 2, 0)) * 255


class SemanticFeed(VisualisationThread):
    def _plot_visualisation(self) -> np.array:
        semantics = self._agent.perception.output_visualisation
        semantics = np.squeeze(np.array(COLOUR_LIST[semantics], dtype=np.uint8))
        return cv2.cvtColor(semantics, cv2.COLOR_RGB2BGR)


class ControlFeed(VisualisationThread):
    def _plot_visualisation(self) -> np.array:
        canvas = get_blank_canvas(self._dimension, self._scale)
        return draw_control_map(self._agent, canvas)


class LocalisationFeed(VisualisationThread):
    def _plot_visualisation(self) -> np.array:
        canvas = get_blank_canvas(self._dimension, self._scale)
        return draw_localisation_map(self._agent, canvas)


MapVisualisationLimit = namedtuple("MapLimit", ["x_min", "x_max", "y_min", "y_max"])
MAP_VISUALISATION_LIMITS = {
    "monza": MapVisualisationLimit(-1100, 300, -1400, 1000),
    "spa": MapVisualisationLimit(-700, 700, -1020, 1120),
    "ks_vallelunga": MapVisualisationLimit(-640, 740, -260, 360),
    "ks_silverstone": MapVisualisationLimit(-560, 560, -900, 900),
    "abudhabi_euroracers_v2": MapVisualisationLimit(-900, 800, -380, 660),
    "ks_nordschleife": MapVisualisationLimit(-3500, 2700, -2500, 4000),
    "rt_bathurst": MapVisualisationLimit(-500, 1040, -700, 1600),
}
MAP_TRANSFORMATION = {
    "monza": flip_and_rotate,
    "spa": flip_and_rotate,
    "ks_vallelunga": flip,
    "ks_silverstone": flip_and_rotate,
    "abudhabi_euroracers_v2": flip,
    "ks_nordschleife": flip,
    "rt_bathurst": flip_and_rotate,
}
ARROW_LENGTH = 25


class MapFeed(VisualisationThread):
    def __init__(self, agent: ElTuarMPC, cfg: Dict, parent=None):
        super().__init__(agent, cfg, parent)
        self._setup_map_feed(cfg)

    def _setup_map_feed(self, cfg: Dict):
        self._track_name = cfg["aci"]["race.ini"]["RACE"]["TRACK"]
        self._map_limits = MAP_VISUALISATION_LIMITS[self._track_name]
        self._map_transform = MAP_TRANSFORMATION[self._track_name]
        self._translation = np.array(
            [-self._map_limits.x_min, -self._map_limits.y_min], dtype=np.int32
        )
        self._setup_map_frame(cfg["mapping"]["map_path"])

    def _setup_map_frame(self, map_path: str):
        self._setup_map_canvas()
        try:
            self._map = load.track_map(map_path)
            self._draw_map()
        except FileNotFoundError:
            self._setup_map_canvas()

    def _setup_map_canvas(self) -> np.array:
        x_extent = self._map_limits.x_max - self._map_limits.x_min
        y_extent = self._map_limits.y_max - self._map_limits.y_min
        self._map_frame = np.zeros((y_extent, x_extent, 3), dtype=np.uint8)

    def _draw_map(self):
        centre_track = self._transform_points(self._map["centre"])
        draw_track_line(self._map_frame, centre_track, (255, 255, 255), 40)
        self._draw_finish_line()

    def _transform_points(self, points: np.array) -> np.array:
        return points.astype(np.int32) + self._translation

    def _draw_finish_line(self):
        finish_line = np.array(
            [
                [self._map["left"][0, 0], self._map["right"][0, 0]],
                [self._map["left"][0, 1], self._map["right"][0, 1]],
            ]
        )
        finish_line = self._transform_points(finish_line)
        draw_track_line(self._map_frame, finish_line, (0, 0, 255), 20)

    def _plot_visualisation(self) -> np.array:
        map_frame = np.copy(self._map_frame)
        self._draw_ego_position(map_frame)
        self._draw_estimated_position(map_frame)
        if self._track_name == "ks_nordschleife":
            map_frame = cv2.resize(map_frame, (0, 0), fx=0.1, fy=0.1)
        map_frame = self._map_transform(map_frame)
        return map_frame

    def _draw_ego_position(self, map_frame: np.array):
        pose = self._agent.game_pose.pose_dict
        # Game uses y up coordinate frame with z forward
        position = np.array([-1.0 * pose["x"], pose["z"]])
        position = self._transform_points(position)
        draw_arrow(
            map_frame,
            position,
            pose["yaw"],
            ARROW_LENGTH,
            (0, 255, 0),
            25,
        )

    def _draw_estimated_position(self, map_frame: np.array):
        if self._agent.localiser is None:
            return
        x, y, yaw = self._agent.localiser.estimated_position
        position = np.array([x, y])
        position = self._transform_points(position)
        draw_arrow(
            map_frame,
            position,
            yaw,
            ARROW_LENGTH,
            (255, 0, 0),
            25,
        )


class VisualisationProvider(QQuickImageProvider):
    imageChanged = pyqtSignal(QImage)

    def __init__(self, feed: FeedThread):
        super().__init__(QQuickImageProvider.ImageType.Image)
        self._feed = feed
        self._feed.updateFrame.connect(self.update_image)
        self._image = None

    def requestImage(self, id, size):
        if self._image:
            image = self._image
        else:
            image = QImage(160, 90, QImage.Format.Format_RGBA8888)
            image.fill(Qt.GlobalColor.black)
        return image, image.size()

    @pyqtSlot(QImage)
    def update_image(self, image: QImage):
        self._image = image
        self.imageChanged.emit(image)

    @pyqtSlot()
    def start(self):
        self._feed.start()

    @pyqtSlot()
    def shutdown(self):
        self._feed.is_running = False
