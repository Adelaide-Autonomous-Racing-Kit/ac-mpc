from __future__ import annotations
import multiprocessing as mp
import time
import threading
import signal
from typing import Dict

import cv2
import numpy as np
from loguru import logger

from visuals import plots

PLOTTING_METHODS = {
    "camera": plots.draw_camera_feed,
    "segmentation": plots.draw_segmentation_map,
    "control_image": plots.draw_control_map,
    "localisation_map": plots.draw_localisation_map,
    "visualised_predictions": plots.draw_visualised_predictions,
}


class Visualiser:
    def __init__(self, agent: ElTuarMPC, cfg: Dict):
        self._visualiser = VisualisationProcess(agent, cfg)
        self._visualiser.start()

    def shutdown(self):
        self._visualiser.is_running = False


class VisualisationProcess(mp.Process):
    def __init__(self, agent: ElTuarMPC, cfg: Dict):
        super().__init__()
        self._agent = agent
        self.__setup(cfg)

    @property
    def is_running(self) -> bool:
        """
        Checks if the localisation process is running

        :return: True if the process is running, false if it is not
        :rtype: bool
        """
        with self._is_running.get_lock():
            is_running = self._is_running.value
        return is_running

    @is_running.setter
    def is_running(self, is_running: bool):
        """
        Sets if the localisation process is running

        :is_running: True if the process is running, false if it is not
        :type is_running: bool
        """
        with self._is_running.get_lock():
            self._is_running.value = is_running

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self._start_display_thread()
        while self.is_running:
            self._draw()
            time.sleep(0.04)

    def _draw(self):
        for name in self._plots_to_draw:
            canvas = self._get_blank_canvas()
            frame = PLOTTING_METHODS[name](self._agent, canvas)
            if frame is not None:
                self._frames_to_display[name] = frame

    def _start_display_thread(self):
        self._plotting_thread = threading.Thread(target=self._display, daemon=True)
        self._plotting_thread.start()

    def _display(self):
        while self.is_running:
            try:
                for window_name in self._frames_to_display.keys():
                    self._show_plot(window_name)
                time.sleep(0.04)
            except Exception as e:
                logger.error(f"Display Thread: {e}")
                self._agent.thread_exception = e

    def _show_plot(self, window_name: str):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, self._frames_to_display[window_name])
        cv2.waitKey(1)

    def _get_blank_canvas(self):
        bev_size = self._birds_eye_view_dimension * self._bev_scale
        return np.zeros((bev_size, bev_size, 3), dtype=np.uint8)

    def __setup(self, cfg: Dict):
        self.__setup_config(cfg)
        self.__setup_shared_memory()

    def __setup_config(self, cfg: Dict):
        self.__setup_canvas_size(cfg)
        self.__setup_plots_to_draw(cfg)

    def __setup_canvas_size(self, cfg: Dict):
        self._birds_eye_view_dimension = cfg["birds_eye_view_size"]
        self._bev_scale = cfg["birds_eye_view_scale"]

    def __setup_plots_to_draw(self, cfg: Dict):
        to_draw = []
        for key in cfg.keys():
            if "show" in key and cfg[key]:
                to_draw.append(key.replace("show_", ""))
        self._plots_to_draw = to_draw
        self._frames_to_display = {}

    def __setup_shared_memory(self):
        self._is_running = mp.Value("i", True)
