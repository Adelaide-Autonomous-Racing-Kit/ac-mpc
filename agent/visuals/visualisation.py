import time
import threading
from typing import Dict

import cv2
import numpy as np

from visuals import plots

PLOTTING_METHODS = {
    "camera": plots.draw_camera_feed,
    "segmentation": plots.draw_segmentation_map,
    "control_image": plots.draw_control_map,
    "localisation_map": plots.draw_localisation_map,
    "visualised_predictions": plots.draw_visualised_predictions,
}


class Visualiser:
    def __init__(self, cfg: Dict, agent):
        self.agent = agent
        self.to_draw = self._get_plots_to_draw(cfg)
        self.birds_eye_view_dimension = cfg["birds_eye_view_size"]
        self.bev_scale = cfg["birds_eye_view_scale"]
        self.frames_to_display = {}
        self._start_display_thread()

    def _get_plots_to_draw(self, cfg: Dict):
        to_draw = []
        for key in cfg.keys():
            if "show" in key and cfg[key]:
                to_draw.append(key.replace("show_", ""))
        return to_draw

    def _start_display_thread(self):
        self.plotting_thread = threading.Thread(target=self.display, daemon=True)
        self.plotting_thread.start()

    def draw(self, obs: Dict):
        for name in self.to_draw:
            canvas = self._get_blank_canvas()
            frame = PLOTTING_METHODS[name](self.agent, canvas, obs)
            if frame is not None:
                self.frames_to_display[name] = frame

    def _get_blank_canvas(self):
        bev_size = self.birds_eye_view_dimension * self.bev_scale
        return np.zeros((bev_size, bev_size, 3), dtype=np.uint8)

    def display(self):
        while True:
            try:
                for window_name in self.frames_to_display.keys():
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(window_name, self.frames_to_display[window_name])
                    cv2.waitKey(1)
                time.sleep(0.5)
            except Exception as e:
                self.agent.thread_exception = e
