import os
import sys

from PyQt6.QtQml import QQmlApplicationEngine, QQmlListProperty
from PyQt6.QtQuick import QQuickWindow
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, pyqtProperty
from PyQt6.QtWidgets import QApplication

from backend.feeds import VisualisationProvider, ThreadCamera

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

VISUALISATION_FEEDS = {
    "cameraFeed": ThreadCamera("/Users/jpb/Desktop/monza-tuna.mp4"),
    "segmentationFeed": ThreadCamera("/Users/jpb/Desktop/monza-cap.mp4"),
    "controlFeed": ThreadCamera("/Users/jpb/Desktop/monza_mpc_localisation_v1.mp4"),
    "predictionsFeed": ThreadCamera("/Users/jpb/Desktop/vallelunga-v1.mp4"),
    "localLocalisationFeed": ThreadCamera("/Users/jpb/Desktop/L2R++@Monza.mp4"),
    "mapLocalisationFeed": ThreadCamera("/Users/jpb/Desktop/AARC-Monza-MPC.mp4"),
}


class Dashboard(QObject):
    pass


def main():
    backend = Dashboard()

    QQuickWindow.setSceneGraphBackend("software")
    app = QApplication(sys.argv)

    engine = QQmlApplicationEngine()
    engine.quit.connect(app.quit)
    engine.rootContext().setContextProperty("backend", backend)
    setup_visualisation_providers(engine)
    engine.load(os.path.join(DIR_PATH, "ui/Main.qml"))

    exit_status = app.exec()
    sys.exit(exit_status)


def setup_visualisation_providers(engine: QQmlApplicationEngine):
    for feed_name in VISUALISATION_FEEDS:
        feed_provider = VisualisationProvider(VISUALISATION_FEEDS[feed_name])
        engine.rootContext().setContextProperty(feed_name, feed_provider)
        engine.addImageProvider(feed_name, feed_provider)


if __name__ == "__main__":
    main()
