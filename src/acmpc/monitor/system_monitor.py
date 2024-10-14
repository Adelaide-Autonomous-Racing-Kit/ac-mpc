from functools import wraps
import time

from aci.utils.system_monitor import SystemMonitor


System_Monitor = SystemMonitor(5000)

def track_runtime(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = function(*args, **kwargs)
        t2 = time.time()
        name = f"{function.__module__}.{function.__name__}"
        System_Monitor.add_function_runtime(name, (t2 - t1) * 10e3)
        return result

    return wrapper