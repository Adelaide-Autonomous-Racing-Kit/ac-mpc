import threading
import time
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import torch
from loguru import logger
from scipy.signal import savgol_filter
from src.interface import AssettoCorsaInterface

from ace.steering import SteeringGeometry
from control.controller import build_mpc
from control.commands import TemporalCommandInterpolator
from localisation.localisation import LocaliseOnTrack
from mapping.map_maker import MapMaker
from monitor.system_monitor import System_Monitor, track_runtime
from perception.perception import Perceiver
from recording.recorder import DataRecorder
from utils import load
from visuals.visualisation import Visualiser

torch.backends.cudnn.benchmark = True


class ElTuarMPC(AssettoCorsaInterface):
    def __init__(self):
        super().__init__()
        self.cfg = load.yaml("agent/configs/params.yaml")
        self.setup()
        self.vehicle_data = SteeringGeometry(self.cfg["vehicle"]["data_path"])
        self.perception = Perceiver(self, self.cfg["perception"], self.cfg["test"])
        self.mapper = MapMaker(verbose=self.cfg["debugging"]["verbose"])
        # self.recorder = DataRecorder(self.save_path, self.cfg["data_collection"])
        self.visualiser = Visualiser(self.cfg["debugging"], self)
        self.localiser = None
        self.localisation_obs = {}
        self.step_count = 0
        System_Monitor.verbosity = self.cfg["debugging"]["verbose"]

    def setup(self):
        self.setup_state()
        self.setup_threading()
        self.seed_packages()

    def setup_state(self):
        self.last_update_timestamp = time.time()
        self.pose = {"velocity": 0.0, "steering_angle": 0.0}
        self.steering_command = 0
        self.acceleration_command = 0
        self.previous_steering_command = 0
        self.previous_acceleration_command = 0
        self.current_time = time.time()
        self._is_mapping_setup = False
        self._is_racing_setup = False

    def setup_threading(self):
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.update_control_lock = threading.Lock()
        self.last_update_timestamp_lock = threading.Lock()
        self.thread_exception = None

    def seed_packages(self):
        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])

    @property
    def centre_track_detections(self) -> np.array:
        return self.tracks["centre"]

    @property
    def left_track_detections(self) -> np.array:
        return self.tracks["left"]

    @property
    def right_track_detections(self) -> np.array:
        return self.tracks["right"]

    @property
    def previous_steering_angle(self) -> float:
        return self.previous_steering_command * self.MPC.delta_max

    @property
    def previous_acceleration(self) -> float:
        acceleration = (
            self.previous_acceleration_command * 16
            if self.previous_acceleration_command < 0
            else self.previous_acceleration_command * 6
        )
        return acceleration

    @property
    def control_input(self) -> np.array:
        with self.last_update_timestamp_lock:
            time_since_update = time.time() - self.last_update_timestamp
        desired_velocity, desired_yaw = self.command_interpolator(time_since_update)
        steering_angle = self._process_yaw(desired_yaw)
        raw_acceleration = self._calculate_acceleration(desired_velocity)
        throttle, brake = self._calculate_commands(raw_acceleration)
        return np.array([steering_angle, brake, throttle])

    def _process_yaw(self, yaw: float) -> float:
        max_steering_angle = self.MPC.delta_max
        steering_angle = -1.0 * np.clip(yaw / max_steering_angle, -1, 1)
        self.steering_command = steering_angle
        return steering_angle

    def _calculate_acceleration(self, desired_velocity: float) -> float:
        max_acceleration = self.MPC.SpeedProfileConstraints["a_max"]
        target = (desired_velocity - self.pose["velocity"]) / 4
        return np.clip(target, -1.0, max_acceleration)

    def _calculate_commands(self, acceleration: float) -> List[float]:
        acceleration = np.clip(acceleration, -1.0, 1.0)
        self.acceleration_command = acceleration
        brake = -1 * acceleration if acceleration < 0 else 0.0
        throttle = acceleration if acceleration > 0 else 0.0
        if self.pose["velocity"] < 10.0:
            throttle = np.clip(throttle, 0.0, 0.60)
        return throttle, brake

    @property
    def control_command(self) -> tuple:
        steering_angle = self.pose["steering_angle"]
        acceleration = self.previous_acceleration
        velocity = self.pose["velocity"]
        return (steering_angle, acceleration, velocity)

    @property
    def reference_path(self) -> np.array:
        ds = int(len(self.centre_track_detections) / self.MPC.MPC_horizon)
        reference_path = np.stack(
            [
                self.centre_track_detections[0::ds, 0],
                self.centre_track_detections[0::ds, 1],
                np.linspace(10.0, 6.0, self.MPC.MPC_horizon),
            ]
        ).T
        return reference_path

    @property
    def reference_speed(self) -> float:
        # reference_speed = self.MPC.v_max
        reference_speed = self.cfg["racing"]["control"]["unlocalised_max_speed"]
        if self.localiser and self.localiser.localised:
            centre_index = self.localiser.estimated_position[1]
            speed_index = centre_index % (len(self.reference_speeds) - 1)
            reference_speed = np.mean(
                self.reference_speeds[speed_index : speed_index + 100]
            )
            logger.info(f"Using reference speed from localisation: {reference_speed}")
        return reference_speed

    def behaviour(self, observation: Dict) -> np.array:
        if self._is_mapping:
            self._maybe_setup_mapping()
            if self._is_mapping_laps_completed(observation):
                return self._finalise_mapping(observation)
        else:
            self._maybe_setup_racing()
        return self.select_action(observation)

    @property
    def _is_mapping(self) -> bool:
        return self.cfg["mapping"]["create_map"] and not self.mapper.map_built

    def _maybe_setup_mapping(self):
        if not self._is_mapping_setup:
            self._setup_mapping()

    def _setup_mapping(self):
        logger.info(
            f"Building a Map from {self.cfg['mapping']['number_of_mapping_laps']} lap"
        )
        self.MPC = build_mpc(self.cfg["mapping"]["control"], self.vehicle_data)
        self.command_interpolator = TemporalCommandInterpolator(self.MPC)
        self._is_mapping_setup = True

    def _is_mapping_laps_completed(self, observation: Dict) -> bool:
        n_laps_completed = observation["state"]["completed_laps"]
        return n_laps_completed >= self.cfg["mapping"]["number_of_mapping_laps"]

    def _finalise_mapping(self, observation: Dict) -> np.array:
        if self._is_car_stopped(observation):
            self._create_map()
        return np.array([0.0, 1.0, 0.0])

    def _is_car_stopped(self, observation) -> bool:
        return observation["state"]["speed_kmh"] <= 1

    def _create_map(self):
        """
        Post-process and save map
        """
        self.mapper.save_map(filename=self.cfg["mapping"]["map_path"])

    def _maybe_setup_racing(self):
        if not self._is_racing_setup:
            self._setup_racing()

    def _setup_racing(self):
        self.MPC = build_mpc(self.cfg["racing"]["control"], self.vehicle_data)
        self.command_interpolator = TemporalCommandInterpolator(self.MPC)
        self._load_model()
        self._is_racing_setup = True

    def select_action(self, obs) -> np.array:
        """
        # Outputs action given the current observation
        returns:
            action: np.array (3,)
            action should be in the form of [\delta, b, t], where \delta is the normalised steering angle,
            t is the normalised throttle and b is normalised brake.
        """
        if self.thread_exception is not None:
            logger.warning(f"Thread Exception Thrown: {self.thread_exception}")

        if self.cfg["multithreading"]:
            self.executor.submit(self.maybe_update_control, obs)
        else:
            self.update_control(obs)
        state = obs["state"]
        speed = np.sqrt(
            state["velocity_x"] ** 2
            + state["velocity_y"] ** 2
            + state["velocity_z"] ** 2
        )
        controls = self.control_input
        System_Monitor.log_select_action(speed)
        return controls

    def maybe_update_control(self, obs):
        if not self.update_control_lock.locked():
            with self.update_control_lock:
                try:
                    self.update_control(obs)
                except Exception as e:
                    self.thread_exception = e
        else:
            logger.warning("Threads queuing - skipping observation")

    def update_control(self, obs):
        self.update_control_state()
        obs = self.perception.perceive(obs)
        self._step(obs)

    def update_control_state(self):
        self.update_time_stamps()
        self.update_previous_control_commands()

    def update_time_stamps(self):
        self.previous_time = self.current_time
        self.current_time = time.time()

    def update_previous_control_commands(self):
        self.previous_steering_command = self.steering_command
        self.previous_acceleration_command = self.acceleration_command

    def _update_state(self, obs: Dict):
        self.dt = self.current_time - self.previous_time
        self.pose["velocity"] = obs["speed"]
        self.pose["steering_angle"] = obs["full_pose"]["SteeringRequest"]
        self.game_pose = (
            obs["full_pose"]["x"],
            obs["full_pose"]["y"],
            obs["full_pose"]["z"],
            obs["full_pose"]["yaw"],
            obs["full_pose"]["pitch"],
            obs["full_pose"]["roll"],
        )
        self.tracks = obs["tracks"]

    def _maybe_add_observations_to_map(self, obs: Dict):
        if not self.mapper.map_built:
            self.mapper.process_segmentation_tracks(
                obs["full_pose"],
                self.left_track_detections,
                self.right_track_detections,
                self.centre_track_detections,
            )

    def _maybe_update_localisation(self):
        if self.localiser:
            self.localiser.step(
                control_command=self.control_command,
                dt=self.dt,
                observation=[self.left_track_detections, self.right_track_detections],
            )
            logger.info(f"Localised: {self.localiser.localised}")
        if self.cfg["localisation"]["collect_benchmark_observations"]:
            localisation_input = {
                "control_command": self.control_command,
                "dt": self.dt,
                "observation": [
                    self.left_track_detections,
                    self.right_track_detections,
                ],
                "game_pose": [self.game_pose],
            }

            self.localisation_obs[self.step_count] = localisation_input

            folder_name = self.cfg["localisation"][
                "benchmark_observations_save_location"
            ]
            np.save(
                f'{folder_name}/{self.cfg["experiment_name"]}.npy',
                self.localisation_obs,
            )

            self.step_count += 1

    def _update_control(self):
        self.update_reference_speed()
        self.MPC.get_control(self.reference_path)
        self.maybe_reset_last_update_timestamp()

    def update_reference_speed(self):
        self.MPC.SpeedProfileConstraints["v_max"] = self.reference_speed
        # reference_speed = self.MPC.SpeedProfileConstraints["v_max"]
        # logger.info(f"Using Reference speed {reference_speed:.2f}")

    def maybe_reset_last_update_timestamp(self):
        if self.MPC.infeasibility_counter == 0:
            with self.last_update_timestamp_lock:
                self.last_update_timestamp = time.time()

    @track_runtime
    def _step(self, obs):
        self._update_state(obs)
        self._maybe_add_observations_to_map(obs)
        self._maybe_update_localisation()
        self._update_control()
        # self._maybe_record_data(obs)
        self._maybe_draw_visualisations(obs)

    def _maybe_record_data(self, obs: Dict):
        self.recorder.maybe_record_data(
            obs,
            self.dt,
            self.steering_angle,
            self.acceleration,
        )

    def _maybe_draw_visualisations(self, obs: Dict):
        self.visualiser.draw(obs)

    def _load_map(self):
        """Loads the generated map"""
        track_dict = np.load(
            self.cfg["mapping"]["map_path"] + ".npy", allow_pickle=True
        ).item()

        tracks = {
            "left": track_dict["outside_track"],
            "right": track_dict["inside_track"],
            "centre": track_dict["centre_track"],
        }
        logger.info(
            f"Loaded map with shapes: {tracks['left'].shape=}"
            + f"{tracks['right'].shape=}, {tracks['centre'].shape=}"
        )
        return tracks

    def register_reset(self, obs) -> np.array:
        """
        Same input/output as select_action, except this method is called at episodal reset.
        Defaults to select_action
        """
        logger.error("RESTART OCCURED (register reset triggered)")
        if self.localiser:
            self.localiser.sampling_strategy()
        return self.select_action(obs)

    def _load_model(self):
        tracks = self._load_map()

        # Remove near duplicate centre points
        d = np.diff(tracks["centre"], axis=0)
        dists = np.hypot(d[:, 0], d[:, 1])
        is_not_duplicated = np.ones(dists.shape[0] + 1).astype(bool)
        is_not_duplicated[1:] = dists > 0.0001
        tracks["left"] = tracks["left"][is_not_duplicated]
        tracks["right"] = tracks["right"][is_not_duplicated]
        tracks["centre"] = tracks["centre"][is_not_duplicated]

        self._calculate_speed_profile(tracks["centre"])
        if self.cfg["localisation"]["use_localisation"]:
            self.localiser = LocaliseOnTrack(
                self.vehicle_data,
                tracks["centre"],
                tracks["left"],
                tracks["right"],
                self.cfg["localisation"],
            )
        self.mapper.map_built = True

    def _calculate_speed_profile(self, centre_track):
        road_width = 9.5
        # TODO: James cringe at this line of code
        centre_track = np.stack(
            [
                centre_track.T[0],
                centre_track.T[1],
                (np.ones(len(centre_track)) * road_width),
            ]
        ).T
        reference_path = self.MPC.construct_waypoints(centre_track)
        for waypoint in reference_path:
            if 0.001 > waypoint["dist_ahead"] > -0.001:
                logger.info(
                    f"Zero Distance Waypoint at: ({waypoint['x']}, {waypoint['y']})"
                )
        reference_path = self.MPC.compute_speed_profile(reference_path)

        plot_ref_path = np.array(
            [[val["x"], val["y"], val["v_ref"]] for i, val in enumerate(reference_path)]
        ).T
        self._plot_speed_profile(plot_ref_path)
        self.reference_speeds = savgol_filter(plot_ref_path[2], 21, 3)

    def _plot_speed_profile(self, ref_path: np.array):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sp = ax.scatter(
            ref_path[0, :],
            ref_path[1, :],
            c=ref_path[2, :] * 3.6,  # / ref_path[2, :].max(),
            cmap=plt.get_cmap("plasma"),
            edgecolor="none",
        )
        ax.set_aspect(1)
        plt.gray()
        fig.colorbar(sp)
        # plt.show()
        plt.savefig("monza_speed.png")
