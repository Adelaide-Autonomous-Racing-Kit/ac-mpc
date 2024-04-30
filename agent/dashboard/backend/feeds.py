from __future__ import annotations
import abc
from typing import Dict

import cv2
import numpy as np
from loguru import logger
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from PyQt6.QtQuick import QQuickImageProvider

from utils import load
from visuals.utils import draw_track_line, draw_arrow
from visuals.plots import (
    COLOUR_LIST,
    draw_control_map,
    draw_localisation_map,
    get_blank_canvas,
)


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
        return self._agent.perception.input_image


class SegmentationFeed(VisualisationThread):
    def _plot_visualisation(self) -> np.array:
        image = self._agent.perception.output_mask
        return np.transpose(image, axes=(1, 2, 0)) * 255


class SemanticFeed(VisualisationThread):
    def _plot_visualisation(self) -> np.array:
        semantics = self._agent.perception.output_visualisation
        semantics = np.squeeze(np.array(COLOUR_LIST[semantics], dtype=np.uint8))
        return semantics


class ControlFeed(VisualisationThread):
    def _plot_visualisation(self) -> np.array:
        canvas = get_blank_canvas(self._dimension, self._scale)
        return draw_control_map(self._agent, canvas)


class LocalisationFeed(VisualisationThread):
    def _plot_visualisation(self) -> np.array:
        canvas = get_blank_canvas(self._dimension, self._scale)
        return draw_localisation_map(self._agent, canvas)


MONZA_X_LIMIT = [-1200, 300]
MONZA_Y_LIMIT = [-1400, 1000]
ARROW_LENGTH = 25


class MapFeed(VisualisationThread):
    def __init__(self, agent: ElTuarMPC, cfg: Dict, parent=None):
        super().__init__(agent, cfg, parent)
        self._setup_map_frame(cfg["mapping"]["map_path"])

    def _setup_map_frame(self, map_path: str):
        self._map = load.track_map(map_path)
        self._map_frame = np.zeros((1500, 2400, 3))
        self._draw_map()

    def _draw_map(self):
        centre_track = self._transform_points(self._map["centre"])

        draw_track_line(self._map_frame, centre_track, (255, 255, 255), 10)
        left_track = self._transform_points(self._map["left"])
        draw_track_line(self._map_frame, left_track, (0, 0, 255), 2)
        right_track = self._transform_points(self._map["right"])
        draw_track_line(self._map_frame, right_track, (0, 0, 255), 2)
        self._draw_finish_line()

    def _transform_points(self, points: np.array) -> np.array:
        return points.astype(np.int32) + np.array([1200, 1400])

    def _draw_finish_line(self):
        finish_line = np.array(
            [
                [self._map["left"][0, 0], self._map["right"][0, 0]],
                [self._map["left"][0, 1], self._map["right"][0, 1]],
            ]
        )
        finish_line = self._transform_points(finish_line)
        draw_track_line(self._map_frame, finish_line, (0, 0, 255), 4)

    def _plot_visualisation(self) -> np.array:
        map_frame = np.copy(self._map_frame)
        self._draw_ego_position(map_frame)
        self._draw_estimated_position(map_frame)
        return map_frame

    def _draw_ego_position(self, map_frame: np.array):
        if self._agent.game_pose is None:
            return
        pose = self._agent.game_pose
        position = np.array([pose["x"], pose["y"]])
        position = self._transform_points(position)
        draw_arrow(
            map_frame,
            position,
            pose["yaw"],
            ARROW_LENGTH,
            (0, 255, 0),
            4,
        )

    def _draw_estimated_position(self, map_frame: np.array):
        x, y, yaw = self._agent.localiser.estimated_position
        position = np.array([x, y])
        position = self._transform_points(position)
        draw_arrow(
            map_frame,
            position,
            yaw,
            ARROW_LENGTH,
            (255, 0, 0),
            4,
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
