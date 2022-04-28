# /usr/bin/python3
"""Profiling decorator"""
import inspect
import os
import threading
import time
from functools import wraps
from queue import Queue
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

from instant_ngp_3dml.utils import DATA_DIR
from instant_ngp_3dml.utils.io import write_json
from instant_ngp_3dml.utils.log import logger

PROFILE = True
PROFILING_T0: float = time.perf_counter()
PROFILING_EVENTS_QUEUE: Queue = Queue()  # [Tuple[str, str, float, float]]


def get_function_name(func: Callable):
    """Get file and function name"""
    module_name = func.__module__.split('.')[-1]
    if module_name == "__main__":
        module_name = os.path.basename(inspect.getfile(func))
    return f"{module_name}::{func.__name__}"


def profile(func: Callable):
    """Profiling decorator"""
    # Warning: This decorator won't work if the function runs inside a multiprocessing Process
    # Processes are not like threads; they do not share memory, which means the global variables are copied and not
    # modified outside the scope of the process
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not PROFILE:
            return func(*args, **kwargs)

        start_time = time.perf_counter()
        retval = func(*args, **kwargs)
        end_time = time.perf_counter()

        # Queue is thread-safe
        PROFILING_EVENTS_QUEUE.put((get_function_name(func), threading.current_thread().name,
                                    start_time-PROFILING_T0, end_time-PROFILING_T0))
        return retval
    return wrapper


def export_profiling_events(output_path: str = os.path.join(DATA_DIR, "profiling.json")):
    """Dump profiling events into a JSON file that can be provided to the Chrome Tracing Viewer"""
    if not PROFILE:
        return

    events: List[Dict[str, Union[str, int, float]]] = []
    while not PROFILING_EVENTS_QUEUE.empty():
        name, tid, t_begin, t_end = PROFILING_EVENTS_QUEUE.get()
        events.append({"name": name, "ph": "B",
                      "ts": t_begin*1e6, "tid": tid, "pid": 0})
        events.append({"name": name, "ph": "E",
                      "ts": t_end*1e6, "tid": tid, "pid": 0})

    write_json(output_path, {"traceEvents": events})
    logger.info(
        f"Open Chrome, type chrome://tracing/ and load the file located at {os.path.abspath(output_path)}")
