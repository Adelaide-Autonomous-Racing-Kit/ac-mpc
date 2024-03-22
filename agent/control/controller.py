from __future__ import annotations
import copy
import multiprocessing as mp
import time
from typing import Dict

import numpy as np
from loguru import logger
from scipy import sparse
from ace.steering import SteeringGeometry

from control.commands import TemporalCommandSelector
from control.dynamics import SpatialBicycleModel
from control.spatial_mpc import SpatialMPC
from perception.shared_memory import SharedPoints


def build_mpc(control_cfg: Dict, vehicle_data):
    Q = sparse.diags(control_cfg["step_cost"])  # e_y, e_psi, t
    R = sparse.diags(control_cfg["r_term"])  # velocity, delta
    QN = sparse.diags(control_cfg["final_cost"])  # e_y, e_psi, t
    v_min = control_cfg["speed_profile_constraints"]["v_min"]
    v_max = control_cfg["speed_profile_constraints"]["v_max"]
    wheel_base = vehicle_data.vehicle_data.wheelbase
    width = vehicle_data.vehicle_data.width
    delta_max = vehicle_data.max_steering_angle()

    InputConstraints = {
        "umin": np.array([v_min, -np.tan(delta_max) / wheel_base]),
        "umax": np.array([v_max, np.tan(delta_max) / wheel_base]),
    }
    StateConstraints = {
        "xmin": np.array([-np.inf, -np.inf, 0.01]),
        "xmax": np.array([np.inf, np.inf, np.inf]),
    }

    model = SpatialBicycleModel(n_states=3, wheel_base=wheel_base, width=width)

    spatial_MPC = SpatialMPC(
        model,
        delta_max,
        control_cfg["horizon"],
        Q,
        R,
        QN,
        StateConstraints,
        InputConstraints,
        copy.copy(control_cfg["speed_profile_constraints"]),
    )
    return spatial_MPC

class Controller:
    def __init__(self, cfg: Dict, perciever: PerceptionProcess):
        self._controller = ControlProcess(cfg, perciever)
        self._controller.start()
    
    @property
    def delta_max(self) -> float:
        return self._controller.model_predictive_controller.delta_max
    
    @property
    def desired_state(self) -> np.array:
        return self._controller.desired_state
    
    @property
    def a_max(self) -> np.array:
        return self._controller.model_predictive_controller.SpeedProfileConstraints["a_max"]
    
    def construct_waypoints(self, path: np.array) -> np.array:
        return self._controller.model_predictive_controller.construct_waypoints(path)

    def compute_speed_profile(self, path: np.array) -> np.array:
        return self._controller.model_predictive_controller.compute_speed_profile(path)


class ControlProcess(mp.Process):
    def __init__(self, cfg: Dict, perciever: PerceptionProcess):
        super().__init__()
        self._perciever = perciever
        self.__setup(cfg)
        self._command_selector = TemporalCommandSelector(self)

    def __setup(self, cfg: Dict):
        self.__setup_config(cfg)
        self.__setup_MPCs()
        self.__setup_shared_memory(cfg)
    
    def __setup_config(self, cfg: Dict):
        self._racing_control_config = cfg["racing"]["control"]
        self._mapping_control_config = cfg["mapping"]["control"]
        self._vehicle_data = SteeringGeometry(cfg["vehicle"]["data_path"])
        self._n_polyfit_points = cfg["perception"]["n_polyfit_points"]

    def __setup_MPCs(self):
        self._mapping_MPC = build_mpc(self._mapping_control_config, self._vehicle_data)
        self._racing_MPC = build_mpc(self._racing_control_config, self._vehicle_data)
    
    def __setup_shared_memory(self, cfg: Dict):
        self.__setup_shared_values()
        self.is_mapping = cfg["mapping"]["create_map"]
        self.__setup_shared_arrays()
    
    def __setup_shared_values(self):
        self._is_running = mp.Value("i", True)
        self._is_mapping = mp.Value("i", True)
        self._shared_update_timestamp = mp.Value("d", 0.0)

    def __setup_shared_arrays(self):
        # TODO: If mapping and racing had different horizons this would cause issues...
        self._shared_control = SharedPoints(self._control_horizon - 1, 2)
        self._shared_cumtime = SharedPoints(self._control_horizon, 0)
        self._shared_predicted_locations = SharedPoints(self._control_horizon - 1, 2)

    @property
    def desired_state(self) -> np.array:
        ret = self._command_selector(self._time_since_update)
        desired_velocity, desired_yaw = self._command_selector(self._time_since_update)
        logger.info(f"Returned: {ret}")
        return desired_velocity, desired_yaw
    
    @property
    def _time_since_update(self) -> float:
        return time.time() - self.last_update_timestamp
    
    @property
    def last_update_timestamp(self) -> float:
        with self._shared_update_timestamp.get_lock():
            timestamp = self._shared_update_timestamp.value
        return timestamp
        
    @last_update_timestamp.setter
    def last_update_timestamp(self, timestamp: float):
        with self._shared_update_timestamp.get_lock():
            self._shared_update_timestamp.value = timestamp
    
    @property
    def control_inputs(self) -> np.array:
        return self._shared_control.points
    
    @control_inputs.setter
    def control_inputs(self, control_inputs: np.array):
        self._shared_control.points = control_inputs

    @property
    def control_cumtime(self) -> np.array:
        return self._shared_cumtime.points
    
    @control_cumtime.setter
    def control_cumtime(self, control_cumtime: np.array):
        self._shared_cumtime.points = control_cumtime

    @property
    def predicted_locations(self) -> np.array:
        return self._shared_predicted_locations.points
        
    @predicted_locations.setter
    def predicted_locations(self, predicted_locations: np.array):
        self._shared_predicted_locations.points = predicted_locations

    @property
    def is_running(self) -> bool:
        """
        Checks if the contol process is running

        :return: True if the contol process is running, false if it is not
        :rtype: bool
        """
        with self._is_running.get_lock():
            is_running = self._is_running.value
        return is_running

    @is_running.setter
    def is_running(self, is_running: bool):
        """
        Sets if the Contol process is running

        :is_running: True if the contol process is running, false if it is not
        :type is_running: bool
        """
        with self._is_running.get_lock():
            self._is_running.value = is_running

    @property
    def is_mapping(self) -> bool:
        """
        Checks if the system is mapping

        :return: True if the system is mapping, false if it is not
        :rtype: bool
        """
        with self._is_mapping.get_lock():
            is_mapping = self._is_mapping.value
        return is_mapping

    @is_mapping.setter
    def is_mapping(self, is_mapping: bool):
        """
        Checks if the system is mapping

        :is_mapping: True if the system is mapping, false if it is not
        :type is_running: bool
        """
        with self._is_mapping.get_lock():
            self._is_mapping.value = is_mapping
    
    @property
    def model_predictive_controller(self) -> SpatialMPC:
        return self._mapping_MPC if self.is_mapping else self._racing_MPC

    def run(self):
        while self.is_running:
            if not self._perciever.is_centreline_stale:
                self._update_control()
    
    def _update_control(self):
        self._update_reference_speed()
        self.model_predictive_controller.get_control(self._reference_path)
        self._update_shared_memory()

    def _update_reference_speed(self):
        self.model_predictive_controller.SpeedProfileConstraints["v_max"] = self._reference_speed

    @property
    def _reference_speed(self) -> float:
        reference_speed = self._racing_control_config["unlocalised_max_speed"]
        # TODO: Add in localisation process
        # if self.localiser and self.localiser.localised:
        #    centre_index = self.localiser.estimated_position[1]
        #    speed_index = centre_index % (len(self.reference_speeds) - 1)
        #    reference_speed = np.mean(
        #        self.reference_speeds[speed_index : speed_index + 100]
        #    )
        #    logger.info(f"Using reference speed from localisation: {reference_speed}")
        return reference_speed

    @property
    def _reference_path(self) -> np.array:
        centreline = self._perciever.centreline
        ds = int(len(centreline) / self._control_horizon)
        centreline = np.stack(
            [
                centreline[0::ds, 0],
                centreline[0::ds, 1],
                np.linspace(10.0, 6.0, self._control_horizon),
            ]
        ).T
        return centreline

    @property
    def _control_horizon(self) -> int:
        mpc = self.model_predictive_controller
        return mpc.MPC_horizon
    
    def _update_shared_memory(self):
        # TODO: Probably lock this entire thing to prevent half updates...
        self.last_update_timestamp = time.time()
        self.control_inputs = self.model_predictive_controller.projected_control.T
        self.control_cumtime = self.model_predictive_controller.cum_time
        self.predicted_locations = self.model_predictive_controller.current_prediction