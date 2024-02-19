from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from localisation.localisation import LocaliseOnTrack
from localisation.tracker import LocalisationTracker


class LocalisationVisualiser:
    def __init__(self, localiser: LocaliseOnTrack, tracker: LocalisationTracker):
        self._particle_filter = localiser
        self._tracker = tracker
        self._subplot_axes = self._setup_plots()

    def _setup_plots(self):
        fig = plt.figure(figsize=(10, 10))
        subfigs = fig.subfigures(3, 1, wspace=0.07)
        top_ax = subfigs[0].subplots(1, 3)
        middle_ax = subfigs[1].subplots(1, 3, sharey=True)
        bottom_ax = subfigs[2].subplots(1, 3, sharey=True)
        axes = {
            "particle_map": top_ax[0],
            "bev_map": top_ax[1],
            "detections": top_ax[2],
            "execution_time": middle_ax[0],
            "distributions": middle_ax[2],
            "error_x": bottom_ax[0],
            "error_y": bottom_ax[1],
            "error_yaw": bottom_ax[2],
        }
        return axes

    def update(self, track_detections: Dict):
        self.plot_localisation_dashboard(track_detections)

    def plot_localisation_dashboard(self, track_detections: Dict):
        for ax in self._subplot_axes.values():
            ax.cla()
        self.plot_particles(self._subplot_axes["particle_map"])
        self.plot_location_pdf(self._subplot_axes["distributions"])
        self.plot_birds_eye_view_map(self._subplot_axes["bev_map"])
        self.plot_local_track(self._subplot_axes["bev_map"], track_detections)
        self.plot_local_track(self._subplot_axes["detections"], track_detections)
        self.plot_location_errors()
        self.plot_execution_time(self._subplot_axes["execution_time"])
        plt.draw()
        plt.pause(0.01)

    @property
    def left_track(self) -> np.array:
        return self._particle_filter.left_track

    @property
    def right_track(self) -> np.array:
        return self._particle_filter.right_track

    @property
    def track(self) -> np.array:
        return self._particle_filter.centre_track

    def plot_particles(self, ax: matplotlib.axes):
        self._setup_particle_map_visualisation_plot(ax)
        self._draw_particles_and_map(ax)

    def _setup_particle_map_visualisation_plot(self, ax: matplotlib.axes):
        ax.set_aspect(1)
        ax.set_xlim(-1200, 300)
        ax.set_ylim(-1400, 1000)
        ax.set_title("Particle filter")

    def _draw_particles_and_map(self, ax: matplotlib.axes):
        self._draw_track_limits(ax)
        self._draw_starting_line(ax)
        self._draw_particles(ax)
        self._draw_ground_truth_position(ax)
        self._draw_estimated_position(ax)
        self._maybe_adjust_plot_limits(ax)

    def _draw_track_limits(self, ax: matplotlib.axes):
        ax.scatter(self.left_track[:, 0], self.left_track[:, 1], c="grey")
        ax.scatter(self.right_track[:, 0], self.right_track[:, 1], c="grey")
        ax.scatter(self.track[:, 0], self.track[:, 1], c="gold")

    def _draw_starting_line(self, ax: matplotlib.axes):
        ax.plot(
            [self.left_track[0, 0], self.right_track[0, 0]],
            [self.left_track[0, 1], self.right_track[0, 1]],
            c="r",
            linewidth=4,
        )

    def _draw_particles(self, ax: matplotlib.axes):
        ax.scatter(
            self._particle_filter.particles["state"][1::4, 0],
            self._particle_filter.particles["state"][1::4, 1],
            c="black",
            s=1,
        )

    def _draw_estimated_position(self, ax: matplotlib.axes):
        arrow_length = 25
        x, y, yaw = self._particle_filter.estimated_position[0]
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        ax.arrow(x, y, dx, dy, width=1, color="r")

    def _draw_ground_truth_position(self, ax: matplotlib.axes):
        arrow_length = 25
        pose = self._tracker.current_ground_truth_pose
        dx = arrow_length * np.cos(pose["yaw"])
        dy = arrow_length * np.sin(pose["yaw"])
        ax.arrow(pose["x"], pose["y"], dx, dy, width=1, color="b")

    def _maybe_adjust_plot_limits(self, ax: matplotlib.axes):
        if self._particle_filter.localised:
            x, y, _ = self._particle_filter.estimated_position[0]
            ax.set_xlim(x - 100, x + 100)
            ax.set_ylim(y - 100, y + 100)

    def plot_location_pdf(self, ax: matplotlib.axes):
        x = self._particle_filter.particles["observation_error"]
        y = self._particle_filter.pdf(np.copy(x)) / self._particle_filter.scale
        ax.plot(x, y, label="offset_pdf")

    # def plot_orientation_pdf():
    #    ax[3].plot(x, orientation_pdf(x) / ori_scale, label="orientation_pdf")

    def plot_location_errors(self):
        self._draw_x_error()
        self._draw_y_error()
        self._draw_yaw_error()

    def _draw_x_error(self):
        ax = self._subplot_axes["error_x"]
        ax.set_title("X Error")
        self._draw_error_histogram(ax, self._tracker._errors["x"])

    def _draw_y_error(self):
        ax = self._subplot_axes["error_y"]
        ax.set_title("Y Error")
        self._draw_error_histogram(ax, self._tracker._errors["y"])

    def _draw_yaw_error(self):
        ax = self._subplot_axes["error_yaw"]
        ax.set_title("Yaw Error")
        self._draw_error_histogram(ax, self._tracker._errors["yaw"])

    def _draw_error_histogram(self, ax: matplotlib.axes, errors: List):
        ax.hist(errors)

    # def plot_orientation_error():

    def plot_birds_eye_view_map(self, ax: matplotlib.axes):
        self._draw_bev_near_best_particle(ax)

    def _draw_bev_near_best_particle(self, ax: matplotlib.axes):
        self._setup_ego_bev_plot(ax)
        self._draw_ego_track(self.left_track, ax, "grey")
        self._draw_ego_track(self.right_track, ax, "grey")
        self._draw_ego_track(self.track, ax, "gold")
        # self._draw_ego_arrow(ax)

    def _setup_ego_bev_plot(self, ax: matplotlib.axes):
        self._setup_default_plot(ax)
        ax.set_aspect(1)
        ax.set_title("Predicted location and track")

    def _setup_default_plot(self, ax: matplotlib.axes):
        size = 200
        x_min, x_max = -size / 2, size / 2
        y_min, y_max = 0, size
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    def _draw_ego_track(self, track: np.array, ax: matplotlib, colour: str):
        track = self._translate_points_to_best_particle(track)
        track = self._rotate_points_to_best_particle(track)
        ax.plot(track[0], track[1], c=colour)

    def _translate_points_to_best_particle(self, points: np.array) -> np.array:
        return np.subtract(points, self._best_particle_position()).T

    def _rotate_points_to_best_particle(self, points: np.array) -> np.array:
        rotation = self._get_best_particle_rotation_transformation()
        return np.matmul(rotation, points)

    def _best_particle_position(self) -> np.array:
        best_particle = np.argmax(self._particle_filter.particles["score"])
        x, y, _ = self._particle_filter.particles["state"][best_particle]
        return np.array([x, y])

    def _best_particle_orientation(self) -> float:
        best_particle = np.argmax(self._particle_filter.particles["score"])
        yaw = self._particle_filter.particles["state"][best_particle][2]
        return yaw

    def _get_best_particle_rotation_transformation(self) -> np.array:
        angle = np.pi / 2 - self._best_particle_orientation()
        return np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        )

    def _draw_ego_arrow(self, ax: matplotlib.axes):
        yaw = self._best_particle_orientation()
        x, y = self._best_particle_position()
        tip = np.array([x + 30 * np.cos(yaw), y + 30 * np.sin(yaw)])
        arrow = np.array([[x, y], tip])
        self._draw_ego_track(arrow, ax, "r")

    def plot_local_track(self, ax: matplotlib.axes, track_detections: Dict):
        self._setup_default_plot(ax)
        self._draw_track_limit_detections(ax, track_detections)
        self._draw_vehicle_pose(ax)

    def _draw_track_limit_detections(
        self, ax: matplotlib.axes, track_detections: np.array
    ):
        left_track, right_track = track_detections["left"], track_detections["right"]
        ax.plot(left_track[:, 0], left_track[:, 1], c="grey")
        ax.plot(right_track[:, 0], right_track[:, 1], c="grey")

    def _draw_vehicle_pose(self, ax: matplotlib.axes):
        ax.arrow(0, 0, 0, 30, width=1, color="g")

    def plot_execution_time(self, ax: matplotlib.axes):
        ax.set_title("Step Time")
        self._draw_error_histogram(ax, self._tracker._execution_times)
