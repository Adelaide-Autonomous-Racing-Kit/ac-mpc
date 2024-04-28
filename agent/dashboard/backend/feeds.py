from __future__ import annotations

import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from PyQt6.QtQuick import QQuickImageProvider


class ThreadCamera(QThread):
    updateFrame = pyqtSignal(QImage)

    def __init__(self, source: str, parent=None):
        QThread.__init__(self, parent)
        self._source = source

    def run(self):
        self.cap = cv2.VideoCapture(self._source)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(
                color_frame.data,
                color_frame.shape[1],
                color_frame.shape[0],
                QImage.Format.Format_RGB888,
            )
            self.updateFrame.emit(img)


class VisualisationProvider(QQuickImageProvider):
    imageChanged = pyqtSignal(QImage)

    def __init__(self, feed: Visualiser):
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
        print("Starting...")
        self._feed.start()

    @pyqtSlot()
    def shutdown(self):
        print("Finishing...")
        try:
            self._feed.cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(e)
