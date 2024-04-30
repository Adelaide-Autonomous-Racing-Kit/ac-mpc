from __future__ import annotations
import os
import sys
import multiprocessing as mp
from typing import Dict

from PyQt6.QtQml import QQmlApplicationEngine
from PyQt6.QtQuick import QQuickWindow
from PyQt6.QtWidgets import QApplication

from dashboard.backend.feeds import (
    CameraFeed,
    ControlFeed,
    LocalisationFeed,
    MapFeed,
    SegmentationFeed,
    SemanticFeed,
    VisualisationProvider,
    VideoThread,
)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

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
    def run(self):
        QQuickWindow.setSceneGraphBackend("software")
        app = QApplication(sys.argv)

        engine = QQmlApplicationEngine()
        engine.quit.connect(app.quit)
        self._setup_visualisation_providers(engine)
        engine.load(os.path.join(DIR_PATH, "ui/Main.qml"))

        exit_status = app.exec()
        sys.exit(exit_status)

    def _setup_visualisation_providers(self, engine: QQmlApplicationEngine):
        for feed_name in VIDEO_FEEDS:
            feed_provider = VisualisationProvider(VIDEO_FEEDS[feed_name])
            engine.rootContext().setContextProperty(feed_name, feed_provider)
            engine.addImageProvider(feed_name, feed_provider)


class DashBoardProcess(TestDashboardProcess):
    def __init__(self, agent: ElTuarMPC, cfg: Dict):
        super().__init__()
        self._agent = agent
        self._cfg = cfg

    def _setup_visualisation_providers(self, engine: QQmlApplicationEngine):
        for feed_name in VISUALISATION_FEEDS:
            feed = VISUALISATION_FEEDS[feed_name](self._agent, self._cfg)
            feed_provider = VisualisationProvider(feed)
            engine.rootContext().setContextProperty(feed_name, feed_provider)
            engine.addImageProvider(feed_name, feed_provider)
