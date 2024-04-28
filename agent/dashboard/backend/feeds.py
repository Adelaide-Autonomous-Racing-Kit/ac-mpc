from __future__ import annotations
import abc
from typing import Dict

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from PyQt6.QtQuick import QQuickImageProvider

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
        self._dimension = cfg["birds_eye_view_size"]
        self._scale = cfg["birds_eye_view_scale"]

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
