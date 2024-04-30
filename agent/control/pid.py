from collections import namedtuple
from typing import Dict

from loguru import logger
from simple_pid import PID

ControlLimits = namedtuple("ControlLimits", ["max", "min"])
CONTROL_LIMITS = {
    "steering": ControlLimits(max=1.0, min=-1.0),
    "throttle":ControlLimits(max=1.0, min=0.0),
    "brake": ControlLimits(max=0.0, min=-1.0),
}

class ControlPID:
    def __init__(self, cfg: Dict):
        self._setup(cfg)

    def __call__(self, current: float, target: float) -> float:
        self._pid.setpoint = target
        control = self._pid(current)
        logger.debug(self._pid.components)
        return control

    def _setup(self, cfg: Dict):
        self._unpack_cfg(cfg)
        self._set_control_limits()
        self._setup_pid()

    def _unpack_cfg(self, cfg: Dict):
        self._sampling_interval = cfg["sampling_interval_s"]
        self._p = cfg["proportional"]
        self._i = cfg["integral"]
        self._d = cfg["derivative"]

    def _set_control_limits(self):
        pass

    def _setup_pid(self):
        self._pid = PID(self._p, self._i, self._d)
        self._pid.sample_time = self._sampling_interval
        self._pid.output_limits = (self._limits.min, self._limits.max)

class SteeringPID(ControlPID):
    def _set_control_limits(self):
        self._limits = CONTROL_LIMITS["steering"]

class ThrottlePID(ControlPID):
    def _set_control_limits(self):
        self._limits = CONTROL_LIMITS["throttle"]

class BrakePID(ControlPID):
    def _set_control_limits(self):
        self._limits = CONTROL_LIMITS["brake"]