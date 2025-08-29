import time
import numpy as np
from typing import Optional
from torch.cuda import nvtx
from contextlib import contextmanager


@contextmanager
def nvtx_range(msg: str):
    """Context manager for NVTX range annotations.

    Args:
        msg (str): Message to be displayed in the NVTX range
    """
    try:
        if is_initialized():
            get_profiler().start(msg)
        nvtx.range_push(msg)
        yield
    finally:
        nvtx.range_pop()
        if is_initialized():
            get_profiler().end(msg)


class Profiler:
    def __init__(self):
        self.dict_nvtx_time = {}
        self.dict_nvtx_start_time = {}
        self.dict_nvtx_end_time = {}

    def start(self, msg: str):
        if msg not in self.dict_nvtx_start_time:
            self.dict_nvtx_start_time[msg] = []
            self.dict_nvtx_end_time[msg] = []
            self.dict_nvtx_time[msg] = []
        self.dict_nvtx_start_time[msg].append(time.time())

    def end(self, msg: str):
        self.dict_nvtx_end_time[msg].append(time.time())
        self.dict_nvtx_time[msg].append(self.dict_nvtx_end_time[msg][-1] - self.dict_nvtx_start_time[msg][-1])

    def get_time(self):
        return self.dict_nvtx_time

    def print_time_distribution(self):
        """
        get the time mean and std of each msg
        """

        for msg, time_list in self.dict_nvtx_time.items():
            print(f"{msg}: len={len(time_list)}, mean={np.mean(time_list)}, std={np.std(time_list)}")

    @staticmethod
    def smart_time_distribution(dict_nvtx_time):
        """ """
        tf = ["Single ", ""]
        components = [
            "comp_prologue",
            "comm_ulysses_1",
            "comp_epilogue",
            "comm_ulysses_2",
            "comp_ring",
        ]
        Request_batches = ["for first Requests_batch", "for second Requests_batch"]

        clean_dict_nvtx_time = {}

        for msg, time_list in dict_nvtx_time.items():
            for tf_ in tf:
                for component in components:
                    for Request_batch in Request_batches:
                        msg_ = f"{tf_}{component} {Request_batch}"
                        if msg_ in msg:
                            if msg_ not in clean_dict_nvtx_time:
                                clean_dict_nvtx_time[msg_] = []
                            clean_dict_nvtx_time[msg_] += time_list

        for msg, time_list in clean_dict_nvtx_time.items():
            print(f"{msg}: len={len(time_list)}, mean={np.mean(time_list)}, std={np.std(time_list)}")

        return clean_dict_nvtx_time


_PROFILER: Optional[Profiler] = None


def is_initialized():
    return _PROFILER is not None


def get_profiler():
    assert is_initialized(), "Profiler has not been initialized."
    return _PROFILER


def initialize_profiler():
    global _PROFILER
    _PROFILER = Profiler()


def clear_profiler():
    _PROFILER = None
    initialize_profiler()


def print_time_distribution():
    return get_profiler().print_time_distribution()


def get_time():
    return get_profiler().get_time()


def smart_time_distribution(dict_nvtx_time):
    return Profiler.smart_time_distribution(dict_nvtx_time)
