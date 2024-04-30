import threading
import time
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt

import numpy as np
import torch
from loguru import logger
from scipy.signal import savgol_filter
from ace.steering import SteeringGeometry
from aci.interface import AssettoCorsaInterface

from control.controller import Controller
from dashboard.dashboard import DashBoardProcess
from localisation.localiser import Localiser
from mapping.map_maker import MapMaker
from monitor.system_monitor import System_Monitor, track_runtime
from perception.perception import Perceiver
from perception.observations import ObservationDict
from utils import load

torch.backends.cudnn.benchmark = True


class ElTuarMPC(AssettoCorsaInterface):
    def __init__(self, config_path: str):
        self.cfg = load.yaml(config_path)
        super().__init__(self.cfg["aci"])
        self.setup()

    @property
    def tracks(self) -> Dict:
        return self.perception.visualisation_tracks

    @property
    def previous_steering_angle(self) -> float:
        return self.previous_steering_command * self.controller.delta_max

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
        desired_velocity, desired_yaw = self.controller.desired_state
        steering_angle = self._process_yaw(desired_yaw)
        raw_acceleration = self._calculate_acceleration(desired_velocity)
        throttle, brake = self._calculate_commands(raw_acceleration)
        return np.array([steering_angle, brake, throttle])

    def _process_yaw(self, yaw: float) -> float:
        max_steering_angle = self.controller.delta_max
        steering_angle = -1.0 * np.clip(yaw / max_steering_angle, -1, 1)
        self.steering_command = steering_angle
        return steering_angle

    def _calculate_acceleration(self, desired_velocity: float) -> float:
        max_acceleration = self.controller.a_max
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
    def reference_speed(self) -> float:
        reference_speed = self.cfg["racing"]["control"]["unlocalised_max_speed"]
        if self._is_using_localisation and self.localiser.is_localised:
            centre_index = self.localiser.estimated_map_index
            speed_index = centre_index % (len(self.reference_speeds) - 1)
            end_index = speed_index + 100
            reference_speed = np.mean(self.reference_speeds[speed_index:end_index])
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
        n_laps = self.cfg["mapping"]["number_of_mapping_laps"]
        logger.info(f"Building a Map from {n_laps} lap")
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
        self._load_model()
        self._is_racing_setup = True

    def select_action(self, obs: List) -> np.array:
        """
        # Outputs action given the current observation
        returns:
            action: np.array (3,)
            action should be in the form of [\delta, b, t], where \delta is the normalised steering angle,
            t is the normalised throttle and b is normalised brake.
        """
        obs = ObservationDict(obs)
        if self.thread_exception is not None:
            logger.warning(f"Thread Exception Thrown: {self.thread_exception}")

        if self.cfg["multithreading"]:
            self.executor.submit(self._maybe_update_control, obs)
        else:
            self._update_control(obs)
        self._step(obs)
        controls = self.control_input
        System_Monitor.log_select_action(obs["speed"])
        return controls

    def _step(self, obs: ObservationDict):
        self._update_control_state()
        self._update_state(obs)
        self._maybe_update_localisation()

    def _maybe_update_control(self, obs):
        if not self.update_control_lock.locked():
            with self.update_control_lock:
                try:
                    self._update_control(obs)
                except Exception as e:
                    self.thread_exception = e

    def _update_control(self, obs):
        self.perception.perceive(obs)
        self._maybe_add_observations_to_map(obs)

    def _update_control_state(self):
        self.update_time_stamps()
        self.update_previous_control_commands()
        self.controller.reference_speed = self.reference_speed

    def update_time_stamps(self):
        self.previous_time = self.current_time
        self.current_time = time.time()

    def update_previous_control_commands(self):
        # TODO: Consider where this should be given threads submitting controls
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

    def _maybe_add_observations_to_map(self, obs: Dict):
        elapsed_time_since_last_update = time.time() - self.last_update_time
        if not self.mapper.map_built and elapsed_time_since_last_update > 0.1:
            tracks = self.tracks
            self.mapper.process_segmentation_tracks(
                obs["full_pose"],
                tracks["left"],
                tracks["right"],
                tracks["centre"],
            )
            self.last_update_time = time.time()

    def _maybe_update_localisation(self):
        if self.localiser:
            self.localiser.step(self.control_command)
        if self._is_collecting_localisation_data:
            # TODO: Move this into localiser.py
            localisation_input = {
                "time": time.time(),
                "control_command": self.control_command,
                "game_pose": [self.game_pose],
            }
            self._localisation_obs[self._step_count] = localisation_input
            self._step_count += 1

    def _load_model(self):
        tracks = load.track_map(self.cfg["mapping"]["map_path"])
        self._calculate_speed_profile(tracks["centre"])
        self.mapper.map_built = True
        if not self.localiser is None:
            self.localiser.start()

    def _calculate_speed_profile(self, centre_track: np.array):
        road_width = 9.5
        # TODO: James cringe at this line of code
        centre_track = np.stack(
            [
                centre_track.T[0],
                centre_track.T[1],
                (np.ones(len(centre_track)) * road_width),
            ]
        ).T
        reference_path = self.controller.construct_waypoints(centre_track)
        reference_path = self.controller.compute_track_speed_profile(reference_path)
        plot_ref_path = np.array(
            [[val["x"], val["y"], val["v_ref"]] for i, val in enumerate(reference_path)]
        ).T
        self._plot_speed_profile(plot_ref_path)
        self.reference_speeds = savgol_filter(plot_ref_path[2], 21, 3)

    def _plot_speed_profile(self, ref_path: np.array):
        fig = plt.figure(dpi=300)
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
        fig.tight_layout()
        plt.savefig("spa_speed.png")
        plt.show()

    def teardown(self):
        self.perception.shutdown()
        self.controller.shutdown()
        if self.localiser:
            self.localiser.shutdown()
        self._maybe_record_localisation_data()
        self.visualiser.terminate()
        # self.visualiser.shutdown()

    def _maybe_record_localisation_data(self):
        if self.localiser and self._is_collecting_localisation_data:
            np.save(self._save_localisation_path, self._localisation_obs)

    def setup(self):
        self._setup_localisation_benchmark_config()
        self._setup_state()
        self._setup_threading()
        self._seed_packages()
        self._setup_vehicle_data()
        self._setup_perception()
        self._setup_controller()
        self._setup_mapper()
        self._setup_controller()
        self._setup_localisation()
        self._setup_monitoring()

    def _setup_localisation_benchmark_config(self):
        cfg = self.cfg["localisation"]
        self._is_using_localisation = cfg["use_localisation"]
        self._is_collecting_localisation_data = cfg["collect_benchmark_observations"]
        save_path = cfg["benchmark_observations_save_location"]
        experiment_name = self.cfg["experiment_name"]
        self._save_localisation_path = f"{save_path}/{experiment_name}/control.npy"

    def _setup_state(self):
        self.game_pose = None
        self.pose = {"velocity": 0.0, "steering_angle": 0.0}
        self.steering_command = 0
        self.acceleration_command = 0
        self.previous_steering_command = 0
        self.previous_acceleration_command = 0
        self.current_time = time.time()
        self._is_mapping_setup = False
        self._is_racing_setup = False
        self.last_update_time = time.time()

    def _setup_threading(self):
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.update_control_lock = threading.Lock()
        self.last_update_timestamp_lock = threading.Lock()
        self.thread_exception = None

    def _seed_packages(self):
        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])

    def _setup_vehicle_data(self):
        self.vehicle_data = SteeringGeometry(self.cfg["vehicle"]["data_path"])

    def _setup_perception(self):
        self.perception = Perceiver(self.cfg["perception"])

    def _setup_controller(self):
        self.controller = Controller(self.cfg, self.perception)

    def _setup_mapper(self):
        self.mapper = MapMaker(verbose=self.cfg["debugging"]["verbose"])

    def _setup_localisation(self):
        if self._is_using_localisation or self._is_collecting_localisation_data:
            self.localiser = Localiser(self.cfg, self.perception)
        else:
            self.localiser = None
        self._localisation_obs = {}
        self._step_count = 0

    def _setup_monitoring(self):
        System_Monitor.verbosity = self.cfg["debugging"]["verbose"]
        self.visualiser = DashBoardProcess(self, self.cfg)
        self.visualiser.start()
