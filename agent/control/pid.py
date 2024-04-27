from collections import namedtuple
from typing import Any, Dict

from simple_pid import PID

ControlLimits = namedtuple("Control Limits", ["max", "min"])
CONTROL_LIMITS = ControlLimits(max=1.0, min=-1.0)


class ControlPID:
    def __init__(self, cfg: Dict):
        self._setup(cfg)

    def __call__(self, current: float, target: float) -> float:
        self._pid.setpoint = target
        return self._pid(current)

    def _setup(self, cfg: Dict):
        self._unpack_cfg(cfg)
        self._setup_control_limits()
        self._setup_pid()

    def _unpack_cfg(self, cfg: Dict):
        self._sampling_interval = self.cfg["sampling_interval_s"]
        self._limits = CONTROL_LIMITS
        self._p = cfg["proportional"]
        self._i = cfg["integral"]
        self._d = cfg["deriviative"]

    def _setup_pid(self):
        self._pid = PID(self._p, self._i, self._d)
        self._pid.sample_time = self._sample_interval
        self._pid.output_limits = (self._limits.min, self._limits.max)
