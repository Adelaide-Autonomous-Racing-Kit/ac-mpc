import pathlib
import threading
import time
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from loguru import logger
from scipy.signal import savgol_filter
from src.interface import AssettoCorsaInterface

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

        self.perception = Perceiver(self, self.cfg["perception"], self.cfg["test"])
        self.MPC = build_mpc(self.cfg["control"], self.cfg["vehicle"])
        self.command_interpolator = TemporalCommandInterpolator(self.MPC)
        self.mapper = MapMaker(verbose=self.cfg["debugging"]["verbose"])
        # self.recorder = DataRecorder(self.save_path, self.cfg["data_collection"])
        self.visualiser = Visualiser(self.cfg["debugging"], self)
        self.localiser = None
        System_Monitor.verbosity = self.cfg["debugging"]["verbose"]

    def setup(self):
        self.setup_state()
        self.setup_threading()
        self.seed_packages()

    def setup_state(self):
        self.last_update_timestamp = time.time()
        self.pose = {"velocity": 0}
        self.steering_command = 0
        self.acceleration_command = 0
        self.previous_steering_command = 0
        self.previous_acceleration_command = 0
        self.current_time = time.time()

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
        return self.previous_steering_command * 0.3

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
        desired_velocity = self._process_velocity(desired_velocity)
        steering_angle = self._process_yaw(desired_yaw)
        raw_acceleration = self._calculate_acceleration(desired_velocity)
        throttle, brake = self._calculate_acceleration_command(raw_acceleration)
        return np.array([steering_angle, brake, throttle])

    def _process_velocity(self, velocity: float) -> float:
        max_velocity = self.cfg["control"]["speed_profile_constraints"]["v_max"]
        min_velocity = self.cfg["control"]["speed_profile_constraints"]["v_min"]
        #return np.clip(velocity, min_velocity, max_velocity)
        return velocity
    
    def _process_yaw(self, yaw: float) -> float:
        max_steering_angle = self.cfg["control"]["input_constraints"]["steering_max"]
        steering_angle = np.clip(yaw / max_steering_angle, -1, 1)
        self.steering_command = steering_angle
        return -1.0 * steering_angle

    def _calculate_acceleration(self, desired_velocity: float) -> float:
        max_acceleration = self.cfg["control"]["speed_profile_constraints"]["a_max"]
        target = (desired_velocity - self.pose["velocity"]) / 4
        logger.info(f"Velocity: {self.pose['velocity']}")
        return np.clip(target, -1.0, max_acceleration)

    def _calculate_acceleration_command(self, acceleration: float) -> List[float]:
        acceleration = np.clip(acceleration, -1.0, 1.0)
        self.acceleration_command = acceleration
        brake = -1 * acceleration if acceleration < 0 else 0.0
        throttle = acceleration if acceleration > 0 else 0.0
        throttle = np.clip(throttle, 0.0, 0.60)
        return throttle, brake

    @property
    def control_command(self) -> tuple:
        steering_angle = self.previous_steering_angle
        acceleration = self.previous_acceleration
        pose = self.pose["velocity"]
        return (steering_angle, acceleration, pose)

    @property
    def reference_path(self) -> np.array:
        ds = int(len(self.centre_track_detections) / self.MPC.MPC_horizon)
        reference_path = np.stack(
            [
                self.centre_track_detections[0::ds, 0],
                self.centre_track_detections[0::ds, 1],
                np.linspace(20.0, 3.0, self.MPC.MPC_horizon),
            ]
        ).T
        return reference_path

    @property
    def reference_speed(self) -> float:
        reference_speed = self.MPC.v_max
        if self.localiser and self.localiser.localised:
            centre_index = self.localiser.estimated_position[1]
            reference_speed = self.reference_speeds[centre_index]
        return reference_speed

    def behaviour(self, observation: Dict) -> np.array:
        return self.select_action(observation)

    def select_action(self, obs) -> np.array:
        """
        # Outputs action given the current observation
        returns:
            action: np.array (2,)
            action should be in the form of [\delta, a], where \delta is the normalized steering angle,
            and a is the normalized acceleration.
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
        logger.info(f"Input Control: {controls}")
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

    def load_map(self, filename: str):
        """Loads the generated map if exists, or the ground truth map otherwise."""
        if self.cfg["mapping"]["create_map"]:
            assert pathlib.Path(filename + ".npy").is_file()
            track_dict = np.load(filename + ".npy", allow_pickle=True).item()
        else:
            track_dict = np.load(
                self.cfg["mapping"]["map_path"] + ".npy", allow_pickle=True
            ).item()

        tracks = {
            "left": track_dict.get("outside_track"),
            "right": track_dict.get("inside_track"),
            "centre": track_dict.get("centre_track"),
        }
        if "raceline" in track_dict:
            self.raceline = track_dict.get("raceline")
        else:
            self.raceline = None
        logger.info(
            f"Loaded map with shapes: {tracks['left'].shape=}"
            + f"{tracks['right'].shape=}, {tracks['centre'].shape=}"
        )
        return tracks

    def training(self, env):
        if not self.cfg["mapping"]["create_map"]:
            return

        logger.info(
            f"Building a Map from {self.cfg['mapping']['number_of_mapping_laps']} lap"
        )
        episode_count = 0

        self.MPC.SpeedProfileConstraints = self.cfg["mapping"][
            "speed_profile_constraints"
        ]
        self.MPC.ay_max = self.MPC.SpeedProfileConstraints["ay_max"]

        while episode_count < self.cfg["mapping"]["number_of_mapping_laps"]:
            state, _ = env.reset()
            self.register_reset(state)

            done = False
            while not done:
                action = self.select_action(state)
                state, reward, done, _ = env.step(action)

            episode_count += 1
            logger.info(f"Completed episode: {episode_count}")

    def register_reset(self, obs) -> np.array:
        """
        Same input/output as select_action, except this method is called at episodal reset.
        Defaults to select_action
        """
        logger.error("RESTART OCCURED (register reset triggered)")
        if self.localiser:
            self.localiser.sampling_strategy()
        return self.select_action(obs)

    def load_model(self, path):
        if pathlib.Path(self.cfg["mapping"]["map_path"]).is_file():
            tracks = self.load_map(filename=self.cfg["mapping"]["map_path"])
        else:
            tracks = self.load_map(filename=path)

        self.MPC.SpeedProfileConstraints = self.cfg["control"][
            "speed_profile_constraints"
        ]
        self.MPC.ay_max = self.MPC.SpeedProfileConstraints["ay_max"]

        self.calculate_speed_profile(tracks["centre"])

        if self.cfg["localisation"]["use_localisation"]:
            self.localiser = LocaliseOnTrack(
                tracks["centre"],
                tracks["left"],
                tracks["right"],
                self.cfg["localisation"],
            )

        self.mapper.map_built = True

    def save_model(self, path):
        """
        Save model checkpoints.
        """
        if self.cfg["mapping"]["create_map"]:
            self.mapper.save_map(filename=path)
            if not pathlib.Path(self.cfg["mapping"]["map_path"]).is_file():
                self.mapper.save_map(filename=self.cfg["mapping"]["map_path"])

    def calculate_speed_profile(self, centre_track):
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
        reference_path = self.MPC.compute_speed_profile(reference_path)

        plot_ref_path = np.array(
            [[val["x"], val["y"], val["v_ref"]] for i, val in enumerate(reference_path)]
        ).T

        self.reference_speeds = savgol_filter(plot_ref_path[2], 21, 3)
