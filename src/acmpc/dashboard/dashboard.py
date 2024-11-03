from __future__ import annotations

import multiprocessing as mp
import os
import sys
from typing import Dict

from PyQt6.QtQml import QQmlApplicationEngine
from PyQt6.QtQuick import QQuickWindow
from PyQt6.QtWidgets import QApplication
from acmpc.dashboard.backend.feeds import (
    CameraFeed,
    ControlFeed,
    LocalisationFeed,
    MapFeed,
    SegmentationFeed,
    SemanticFeed,
    VideoThread,
    VisualisationProvider,
)
from acmpc.dashboard.backend.session_information import SessionInformationProvider

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

UI_PATHS = {
    "dashboard": os.path.join(DIR_PATH, "ui/Main.qml"),
    "streaming": os.path.join(DIR_PATH, "ui/MainStream.qml"),
}
VIDEO_FEEDS = {
    "cameraFeed": VideoThread("/Users/jpb/Desktop/monza-tuna.mp4"),
    "segmentationFeed": VideoThread("/Users/jpb/Desktop/monza-cap.mp4"),
    "controlFeed": VideoThread("/Users/jpb/Desktop/monza_mpc_localisation_v1.mp4"),
    "predictionsFeed": VideoThread("/Users/jpb/Desktop/vallelunga-v1.mp4"),
    "localLocalisationFeed": VideoThread("/Users/jpb/Desktop/L2R++@Monza.mp4"),
    "mapLocalisationFeed": VideoThread("/Users/jpb/Desktop/AARC-Monza-MPC.mp4"),
}
VISUALISATION_FEEDS = {
    "cameraFeed": CameraFeed,
    "segmentationFeed": SegmentationFeed,
    "controlFeed": ControlFeed,
    "predictionsFeed": SemanticFeed,
    "localLocalisationFeed": LocalisationFeed,
    "mapLocalisationFeed": MapFeed,
}


class TestDashboardProcess(mp.Process):
    def __init__(self):
        super().__init__()
        self._ui_path = UI_PATHS["dashboard"]

    def run(self):
        QQuickWindow.setSceneGraphBackend("software")
        self._app = QApplication(sys.argv)

        engine = QQmlApplicationEngine()
        engine.quit.connect(self._app.quit)
        self._setup_providers(engine)
        engine.load(self._ui_path)
        self._exit_status = self._app.exec()

    def shutdown(self):
        self.terminate()
        self.join()

    def _setup_providers(self, engine: QQmlApplicationEngine):
        self._setup_visualisation_providers(engine)

    def _setup_visualisation_providers(self, engine: QQmlApplicationEngine):
        for feed_name in VIDEO_FEEDS:
            feed_provider = VisualisationProvider(VIDEO_FEEDS[feed_name])
            engine.rootContext().setContextProperty(feed_name, feed_provider)
            engine.addImageProvider(feed_name, feed_provider)


class DashBoardProcess(TestDashboardProcess):
    def __init__(self, agent: ElTuarMPC, cfg: Dict):
        super().__init__()
        self.daemon = True
        self._agent = agent
        self._cfg = cfg
        if self._is_streaming:
            self._ui_path = UI_PATHS["streaming"]
        else:
            self._ui_path = UI_PATHS["dashboard"]

    @property
    def _is_streaming(self) -> bool:
        return self._cfg["is_streaming"]

    def _setup_providers(self, engine: QQmlApplicationEngine):
        self._setup_visualisation_providers(engine)
        self._setup_session_information_provider(engine)

    def _setup_visualisation_providers(self, engine: QQmlApplicationEngine):
        for feed_name in VISUALISATION_FEEDS:
            feed = VISUALISATION_FEEDS[feed_name](self._agent, self._cfg)
            feed_provider = VisualisationProvider(feed)
            engine.rootContext().setContextProperty(feed_name, feed_provider)
            engine.addImageProvider(feed_name, feed_provider)

    def _setup_session_information_provider(self, engine: QQmlApplicationEngine):
        self._session_info = SessionInformationProvider(self._agent)
        engine.rootContext().setContextProperty("sessionInfo", self._session_info)
