from concurrent.futures import ThreadPoolExecutor
import threading
import time
from typing import Dict, Tuple

from ace.steering import SteeringGeometry
from aci.interface import AssettoCorsaInterface
from aci.utils.system_monitor import SystemMonitor, track_runtime
from acmpc.control.controller import Controller
from acmpc.control.pid import BrakePID, SteeringPID, ThrottlePID
from acmpc.dashboard.dashboard import DashBoardProcess
from acmpc.localisation.localiser import Localiser
from acmpc.mapping.map_maker import MapMaker
from acmpc.perception.observations import ObservationDict
from acmpc.perception.perception import Perceiver
from acmpc.state.shared_memory import SharedPose, SharedSessionDetails
from acmpc.utils import load
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import torch

MINIMUM_PROGRESS_M = 50
MINIMUM_PROGRESS = 0.0005
MINIMUM_FUEL_L = 0.01
REFERENCE_SPEED_WINDOW_AHEAD = 75
REFERENCE_SPEED_WINDOW_BEHIND = 25

Agent_Monitor = SystemMonitor(10000)


class ElTuarMPC(AssettoCorsaInterface):
    def __init__(self, config_path: str):
        self.cfg = load.yaml(config_path)
        super().__init__(self.cfg["aci"])
        self.setup()
        
    def restart_condition(self, observation: Dict) -> bool:
        return False

    def termination_condition(self, observation: Dict) -> bool:
        """
        Terminates the run if an agent does not make any progress around the track
        """
        is_not_progressing = not self._is_progressing(observation)
        is_out_of_fuel = self._is_out_of_fuel(observation)
        if is_not_progressing:
            logger.warning("Agent is not making progress")
        if is_out_of_fuel:
            logger.warning("Agent is out of fuel")
        return is_not_progressing or is_out_of_fuel

    def _is_progressing_distance(self, observation: Dict) -> bool:
        is_progressing = True
        current_distance = observation["state"]["distance_traveled"]
        if self._previous_distance is not None:
            distance = abs(current_distance - self._previous_distance)
            if distance < MINIMUM_PROGRESS_M:
                is_progressing = False
        self._previous_distance = current_distance
        return is_progressing

    def _is_out_of_fuel(self, observation: Dict) -> bool:
        remaining_fuel = observation["state"]["fuel"]
        return remaining_fuel < MINIMUM_FUEL_L

    def _is_progressing(self, observation: Dict) -> bool:
        is_progressing = True
        current_position = observation["state"]["normalised_car_position"]
        if self._previous_position is not None:
            distance = abs(current_position - self._previous_position)
            if distance < MINIMUM_PROGRESS:
                is_progressing = False
        self._previous_position = current_position
        return is_progressing

    @property
    def is_localised(self) -> bool:
        return self._is_using_localisation and self.localiser.is_localised

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
        throttle, brake = self._calculate_acceleration(desired_velocity)
        return np.array([steering_angle, brake, throttle])

    def _process_yaw(self, yaw: float) -> float:
        max_steering_angle = self.controller.delta_max
        target_steering_angle = -1.0 * np.clip(yaw / max_steering_angle, -1, 1)
        current_steering_angle = self.pose["steering_angle"]
        delta_steering_angle = self._steering_pid(
            current_steering_angle, target_steering_angle
        )
        steering_angle = current_steering_angle + delta_steering_angle
        self.steering_command = steering_angle
        return steering_angle

    def _calculate_acceleration(self, target_velocity: float) -> Tuple[float, float]:
        current_velocity = self.pose["velocity"]
        throttle = self._throttle_pid(current_velocity, target_velocity)
        brake = self._brake_pid(current_velocity, target_velocity)
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
        if self.is_localised:
            reference_speed = self._reference_speed
        return reference_speed

    @property
    def _reference_speed(self) -> float:
        centre_index = self.localiser.estimated_map_index
        start = centre_index - REFERENCE_SPEED_WINDOW_BEHIND
        end = centre_index + REFERENCE_SPEED_WINDOW_AHEAD
        indices = np.arange(start, end)
        return np.mean(self.reference_speeds.take(indices, mode="wrap"))

    def behaviour(self, observation: Dict) -> np.array:
        if self._is_mapping:
            self._maybe_setup_mapping()
            if self._is_mapping_laps_completed(observation):
                return self._finalise_mapping(observation)
        else:
            self._maybe_setup_racing()
        # Agent_Monitor.maybe_log_function_itterations_per_second()
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

    @track_runtime(Agent_Monitor)
    def select_action(self, obs: Dict) -> np.array:
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
        self.controller.is_localised = self.is_localised

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
        self.game_pose.pose = obs
        self.session_info.session_info = obs

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
                "game_pose": [self.game_pose.pose],
            }
            self._localisation_obs[self._step_count] = localisation_input
            self._step_count += 1

    def _load_model(self):
        tracks = load.track_map(self.cfg["mapping"]["map_path"])
        self._calculate_speed_profile(tracks["centre"])
        self.mapper.map_built = True
        if self.localiser is not None:
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
        reference_path = self.controller.compute_track_speed_profile(centre_track)
        plot_ref_path = np.array(
            [reference_path.xs, reference_path.ys, reference_path.velocities]
        )
        self._plot_speed_profile(plot_ref_path)
        self.reference_speeds = savgol_filter(plot_ref_path[2], 21, 3)

    def _plot_speed_profile(self, ref_path: np.array):
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        sp = ax.scatter(
            ref_path[0, :],
            ref_path[1, :],
            c=ref_path[2, :] * 3.6,
            cmap=plt.get_cmap("plasma"),
            edgecolor="none",
        )
        ax.set_aspect(1)
        plt.gray()
        fig.colorbar(sp)
        fig.tight_layout()
        filename = f"{self._experiment_name}_speed.png"
        output_path = f"{self._save_speed_profile_path}/{filename}"
        plt.savefig(output_path)
        # plt.show()

    def teardown(self):
        self.perception.shutdown()
        self.controller.shutdown()
        if self.localiser:
            self.localiser.shutdown()
        self._maybe_record_localisation_data()
        self.visualiser.shutdown()

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
        self._setup_localisation()
        self._setup_monitoring()

    def _setup_localisation_benchmark_config(self):
        cfg = self.cfg["localisation"]
        self._is_using_localisation = cfg["use_localisation"]
        self._is_collecting_localisation_data = cfg["collect_benchmark_observations"]
        save_path = cfg["benchmark_observations_save_location"]
        self._experiment_name = self.cfg["experiment_name"]
        self._save_localisation_path = (
            f"{save_path}/{self._experiment_name}/control.npy"
        )

    def _setup_state(self):
        self.game_pose = SharedPose()
        self.session_info = SharedSessionDetails()
        self.pose = {"velocity": 0.0, "steering_angle": 0.0}
        self.steering_command = 0
        self.acceleration_command = 0
        self.previous_steering_command = 0
        self.previous_acceleration_command = 0
        self.current_time = time.time()
        self._is_mapping_setup = False
        self._is_racing_setup = False
        self.last_update_time = time.time()
        self._previous_distance = None
        self._previous_position = None

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
        pid_cfg = self.cfg["acceleration_pid"]
        self._throttle_pid = ThrottlePID(pid_cfg["throttle_pid"])
        self._brake_pid = BrakePID(pid_cfg["brake_pid"])
        self._steering_pid = SteeringPID(self.cfg["steering_pid"])

    def _setup_mapper(self):
        self.mapper = MapMaker(verbose=self.cfg["debugging"]["verbose"])
        self._save_speed_profile_path = self.cfg["debugging"]["speed_profile_path"]

    def _setup_localisation(self):
        if self._is_using_localisation or self._is_collecting_localisation_data:
            self.localiser = Localiser(self.cfg, self.perception)
        else:
            self.localiser = None
        self._localisation_obs = {}
        self._step_count = 0

    def _setup_monitoring(self):
        # System_Monitor.verbosity = self.cfg["debugging"]["verbose"]
        self.visualiser = DashBoardProcess(self, self.cfg)
        self.visualiser.start()
        self._last_image_timestamp = time.time()
