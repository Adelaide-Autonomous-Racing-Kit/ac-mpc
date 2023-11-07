import time
from collections import defaultdict
from functools import wraps

import numpy as np
from loguru import logger


def track_runtime(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = function(*args, **kwargs)
        t2 = time.time()
        name = f"{function.__module__}.{function.__name__}"
        System_Monitor.add_function_runtime(name, t2 - t1)
        return result

    return wrapper


class SystemMonitor:
    def __init__(self):
        self.runtimes = defaultdict(list)
        self.start_time, self.speeds = -1, []
        self.select_action_count, self.action_update_count = 0, 0

    @property
    def average_speed(self) -> float:
        return np.mean(self.speeds)

    @property
    def select_actions_per_second(self) -> float:
        time_elapsed = time.time() - self.start_time
        return self.select_action_count / time_elapsed

    def add_function_runtime(self, function_name: str, runtime: float):
        self.runtimes[function_name].append(runtime)

    def average_runtime(self, function_name: str) -> float:
        return np.mean(self.runtimes[function_name])

    def log_select_action(self, speed: float):
        self.maybe_timestamp_start()
        self.update_select_action_stats(speed)
        self.maybe_log_select_action_summary()
        self.maybe_stop_for_runtime_issues()

    def maybe_timestamp_start(self):
        if self.start_time == -1:
            self.start_time = time.time()
            self.step_count = 1

    def update_select_action_stats(self, speed: float):
        self.select_action_count += 1
        self.speeds.append(speed)

    def maybe_stop_for_runtime_issues(self):
        self.stop_if_processing_is_slow()
        self.stop_if_progressing_too_slowly()

    def maybe_log_select_action_summary(self):
        if self.is_verbose() and self.is_logging_interval():
            self.log_processing_time()
            self.log_vehicle_speed
            self.log_function_runtimes_times()

    def log_processing_time(self):
        message = "Processing frequency: "
        message += f"{self.select_actions_per_second:.2f} it/s"
        logger.info(message)

    def log_vehicle_speed(self):
        message = f"Average speed: {self.average_speed:.2f}m/s"
        message += f" = {self.average_speed*3.6:.2f}km/h"
        logger.info(message)

    def log_function_runtimes_times(self):
        for key in self.runtimes.keys():
            average_runtime = self.average_runtime(key)
            logger.info(f"{key} runs: {average_runtime:.4f}s")

    def stop_if_processing_is_slow(self):
        if self.is_processing_slowly() and not self.cfg["evaluation"]:
            message = "Not running fast enough"
            message += f"{self.select_actions_per_second:.2f}it/s)"
            logger.error(message)
            raise Exception("Processing too slow, ABORT")

    def stop_if_progressing_too_slowly(self):
        if self.is_progressing_slowly() and not self.cfg["evaluation"]:
            message = "Car travelling too slow to set decent time:"
            message += f"({self.average_speed:.2f}m/s)"
            logger.error(message)
            raise Exception("Vehicle travelling too slow, ABORT")

    def is_verbose(self) -> bool:
        return self.verbosity

    def is_progressing_slowly(self) -> bool:
        return self.step_count > 200 and self.average_speed < 10

    def is_processing_slowly(self) -> bool:
        return self.step_count > 10 and self.select_actions_per_second < 2.0

    def is_logging_interval(self) -> bool:
        return self.select_action_count % 30 == 0


System_Monitor = SystemMonitor()
