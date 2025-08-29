# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Megatron timers with energy monitoring capabilities.

This module provides hierarchical timing and energy monitoring for Megatron training.
It supports both basic timing functionality and advanced energy consumption tracking
using NVIDIA Management Library (NVML).

Key Features:
- Hierarchical timer management with log levels
- CUDA-synchronized timing for accurate GPU measurements  
- Optional energy consumption monitoring per timer
- Background energy polling process for continuous monitoring
- CSV export for timing and energy data
- Distributed training support with barrier synchronization

Usage Examples:

Basic Usage (existing functionality):
    timers = Timers(log_level=1, log_option='max')
    
    forward_timer = timers('forward_pass', log_level=0)
    forward_timer.start()
    # ... forward pass code ...
    forward_timer.stop()
    
    timers.log(['forward_pass'])

Enhanced Usage with Energy Monitoring:
    timers = Timers(
        log_level=1, 
        log_option='max',
        device_idx=0,  # GPU 0
        enable_energy_monitoring=True,
        output_dir='./logs'
    )
    
    compute_timer = timers('compute', log_level=0)
    compute_timer.start()
    # ... computation code ...
    compute_timer.stop()
    
    # Log with energy information
    timers.log_with_energy(['compute'])
    
    # Export data and cleanup
    timers.shutdown()

Log Levels:
- Level 0: Critical/always-on timers (highest priority)
- Level 1: Moderate importance timers  
- Level 2: Debug/detailed timers (lowest priority)

Energy Monitoring Requirements:
- Requires pynvml package: pip install pynvml
- Requires NVIDIA GPU with energy monitoring support
- May require elevated privileges on some systems
"""

import time
from abc import ABC, abstractmethod
from typing import List
import multiprocessing as mp
import pynvml
import os

import torch

from megatron.core.utils import is_torch_min_version

if is_torch_min_version("1.13.0"):
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
else:
    dist_all_gather_func = torch.distributed._all_gather_base


class TimerBase(ABC):
    """Timer base class."""

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def start(self, barrier=False):
        """Start the timer.

        Args:
            barrier (bool, optional): Synchronizes ranks before starting. Defaults to False.
        """
        pass

    @abstractmethod
    def stop(self, barrier=False):
        """Stop the timer.

        Args:
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset timer."""
        pass

    @abstractmethod
    def elapsed(self, reset=True, barrier=False):
        """Calculates the elapsed time and restarts timer.

        Args:
            reset (bool, optional): Resets timer before restarting. Defaults to True.
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.

        Returns:
            float: Elapsed time.
        """
        pass


class DummyTimer(TimerBase):
    """Dummy Timer."""

    def __init__(self):
        super().__init__('dummy timer')

    def start(self, barrier=False):
        return

    def stop(self, barrier=False):
        return

    def reset(self):
        return

    def elapsed(self, reset=True, barrier=False):
        raise Exception(
            'dummy timer should not be used to calculate elapsed time, '
            'check if timer\'s log_level <= self._log_level.'
        )

    def active_time(self):
        """Returns the cumulative duration the timer has been active.
        Note: Not supported for DummyTimer.
        """
        raise Exception(
            'active timer should not be used to calculate elapsed time, '
            'check if timer\'s log_level <= self._log_level.'
        )


class Timer(TimerBase):
    """
    Timer class with ability to start/stop.

    Comment on using `barrier`: If this flag is passed, then all
    the caller processes will wait till all reach the timing routine.
    It is up to the user to make sure all the ranks in `barrier_group`
    call it otherwise, it will result in a hang.
    Comment on `barrier_group`: By default it is set to None which
    in torch distributed land, it will result in the global communicator.
    """

    def __init__(self, name, device_idx=None, enable_energy_monitoring=False):
        """Initialize Timer.

        Args:
            name (str): Name of the timer.
            device_idx (int, optional): GPU device index for energy monitoring.
            enable_energy_monitoring (bool, optional): Enable energy monitoring.
        """
        super().__init__(name)
        self._elapsed = 0.0
        self._active_time = 0.0
        self._started = False
        # Note that None will default to the global process group
        self._barrier_group = None
        self._start_time = time.time()
        
        # Track individual start/end times for CSV export
        self.start_timings = []
        self.end_timings = []
        
        # Energy monitoring attributes
        self._enable_energy_monitoring = enable_energy_monitoring
        self._gpu_handle = None
        self._start_energy = 0.0
        self._energy_consumed = []
        self._energy_total = 0.0
        
        # if self._enable_energy_monitoring:
        #     assert device_idx is not None, "device_idx must be provided for energy monitoring"
        #     pynvml.nvmlInit()
        #     self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def set_barrier_group(self, barrier_group):
        """Sets barrier group.

        Args:
            barrier_group (ProcessGroup): Torch ProcessGroup for barrier.
        """
        self._barrier_group = barrier_group

    def start(self, barrier=False):
        """Start the timer.

        Args:
            barrier (bool, optional): Synchronizes ranks before starting. Defaults to False.
        """
        assert not self._started, 'timer has already been started'
        if barrier:
            torch.distributed.barrier(group=self._barrier_group)
        torch.cuda.synchronize()
        
        # if self._enable_energy_monitoring and self._gpu_handle:
        #     self._start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._gpu_handle) / 1000.0
        
        self._start_time = time.time()
        self.start_timings.append(self._start_time)  # Record for CSV export
        self._started = True

    def stop(self, barrier=False):
        """Stop the timer.

        Args:
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.
        """
        assert self._started, 'timer is not started'
        if barrier:
            torch.distributed.barrier(group=self._barrier_group)
        torch.cuda.synchronize()
        
        end_time = time.time()
        self.end_timings.append(end_time)  # Record for CSV export
        elapsed = end_time - self._start_time
        self._elapsed += elapsed
        self._active_time += elapsed
        
        # Record energy consumption
        # if self._enable_energy_monitoring and self._gpu_handle:
        #     end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._gpu_handle) / 1000.0
        #     energy_diff = end_energy - self._start_energy
        #     self._energy_consumed.append(energy_diff)
        #     self._energy_total += energy_diff
        
        self._started = False

    def reset(self):
        """Reset timer."""
        # Don't reset _active_time or _energy_total (cumulative metrics)
        self._elapsed = 0.0
        self._started = False
        self._energy_consumed = []
        self.start_timings = []
        self.end_timings = []

    def elapsed(self, reset=True, barrier=False):
        """Calculates the elapsed time and restarts timer.

        Args:
            reset (bool, optional): Resets timer before restarting. Defaults to True.
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.

        Returns:
            float: Elapsed time.
        """
        _started = self._started
        # If the timing in progress, end it first.
        if self._started:
            self.stop(barrier=barrier)
        # Get the elapsed time.
        _elapsed = self._elapsed
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if _started:
            self.start(barrier=barrier)
        return _elapsed

    def active_time(self):
        """Calculates the cumulative duration for which the timer has been active"""
        return self._active_time

    def energy_consumed(self):
        """Returns the total energy consumed by this timer in Joules."""
        return self._energy_total

    def energy_measurements(self):
        """Returns list of individual energy measurements in Joules."""
        return self._energy_consumed.copy()


def energy_polling_process(device_idx: int, channel: mp.SimpleQueue, output_dir: str, rank: int) -> None:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    timings = []
    energy_consumed = []
    prev_energy = -1.0
    i = 0
    while True:
        timing = time.time()
        energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle) / 1000.0
        if prev_energy != energy:
            timings.append(timing)
            energy_consumed.append(energy)
            prev_energy = energy

        i += 1
        if not channel.empty():
            print(f"energy polling process for rank {rank} received stop signal")
            break

    pynvml.nvmlShutdown()

    with open(f"{output_dir}/time-energy-{rank}.csv", "w") as f:
        f.write("time,energy\n")
        for timing, energy in zip(timings, energy_consumed):
            f.write(f"{timing},{energy}\n")


class Timers:
    """Class for a group of Timers with optional energy monitoring."""

    def __init__(self, log_level, log_option, device_idx=None, enable_energy_monitoring=False, output_dir=None):
        """Initialize group of timers.

        Args:
            log_level (int): Log level to control what timers are enabled.
            log_option (str): Setting for logging statistics over ranks for all the timers.
                              Allowed: ['max', 'minmax', 'all'].
            device_idx (int, optional): GPU device index for energy monitoring.
            enable_energy_monitoring (bool, optional): Enable energy monitoring.
            output_dir (str, optional): Directory to save energy/timing CSV files.
        """
        self._log_level = log_level
        allowed_log_options = set(['max', 'minmax', 'all'])
        assert (
            log_option in allowed_log_options
        ), 'input log option {} is invalid. It must be one of {}'.format(
            log_option, allowed_log_options
        )
        self._log_option = log_option
        self._timers = {}
        self._log_levels = {}
        self._dummy_timer = DummyTimer()
        self._max_log_level = 2
        
        # Energy monitoring attributes
        self._device_idx = device_idx
        self._enable_energy_monitoring = enable_energy_monitoring
        self._output_dir = output_dir
        self._energy_polling_channel = None
        self._energy_polling_process = None

        # Start energy polling process if enabled
        if self._enable_energy_monitoring:
            assert self._device_idx is not None, "device_idx must be provided for energy monitoring"
            assert self._output_dir is not None, "output_dir must be provided for energy monitoring"
            assert self._device_idx is not None, "device_idx must be provided for energy monitoring"
            os.makedirs(self._output_dir, exist_ok=True)
            self._start_energy_polling()

    def _start_energy_polling(self):
        """Start the background energy polling process."""
        print(f"Starting energy polling process for rank {self._device_idx}")
        self._energy_polling_channel = mp.SimpleQueue()
        self._energy_polling_process = mp.Process(
            target=energy_polling_process,
            args=(self._device_idx, self._energy_polling_channel, self._output_dir, self._device_idx)
        )
        self._energy_polling_process.start()
        time.sleep(1)

    def _stop_energy_polling(self):
        """Stop the background energy polling process."""
        if self._energy_polling_process and self._energy_polling_channel:
            self._energy_polling_channel.put("end")
            print(f"Stopping energy polling process for rank {self._device_idx}")
            self._energy_polling_process.join(timeout=5.0)
            if self._energy_polling_process.is_alive():
                self._energy_polling_process.terminate()
                self._energy_polling_process.join()

    def __del__(self):
        """Cleanup energy polling process on destruction."""
        self._stop_energy_polling()

    def shutdown(self):
        """Explicitly shutdown energy monitoring and export data."""
        self._stop_energy_polling()
        self._export_timing_data()

    def __call__(self, name, log_level=None):
        """Call timer with name and log level."""
        # If the timer has already been set, then check if the log-level
        # is provided, it matches the one that the timer was created with.
        if name in self._timers:
            if log_level is not None:
                assert log_level == self._log_levels[name], (
                    'input log level {} does not match already existing '
                    'log level {} for {} timer'.format(log_level, self._log_levels[name], name)
                )
            return self._timers[name]
        # If timer does not exist and no log level is provided,
        # set it to the max log level which is 2.
        if log_level is None:
            log_level = self._max_log_level
        assert (
            log_level <= self._max_log_level
        ), 'log level {} is larger than max supported log level {}'.format(
            log_level, self._max_log_level
        )
        # Now if the input log level is larger than the one set for
        # the timers class, just ignore it and return a dummy timer.
        if log_level > self._log_level:
            return self._dummy_timer
        # Otherwise, initalize the timer and set the level.
        self._timers[name] = Timer(
            name, 
            device_idx=self._device_idx, 
            enable_energy_monitoring=self._enable_energy_monitoring
        )
        self._log_levels[name] = log_level
        return self._timers[name]

    def _export_timing_data(self):
        """Export timing and energy data to CSV files."""
        if not self._output_dir:
            return
        
        # Export timing data
        with open(f"{self._output_dir}/instructions-{self._device_idx}.csv", "w") as f:
            f.write("instruction,start,end\n")
            for name, timer in self._timers.items():
                if isinstance(timer, Timer):  # Skip DummyTimer
                    for start, end in zip(timer.start_timings, timer.end_timings):
                        f.write(f"{name},{start},{end}\n")

    def _get_elapsed_time_all_ranks(self, names, reset, barrier):
        """Returns elapsed times of timers in names.
        Assumptions:
            - All the ranks call this function.
            - `names` are identical on all ranks.
        If the above assumptions are not met, calling this function will
        result in hang.

        Args:
            names (List[str]): list of timer names
            reset (bool): reset the timer after recording the elapsed time
            barrier (bool): if set, do a global barrier before time measurments

        Returns:
            torch.tensor: Tensor of size [world_size, len(names)] with times in float.
        """

        # First make sure all the callers are in sync.
        if barrier:
            torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        # Here we can use gather on the rank we want to print the
        # timing, however, there is no gather_base support in
        # pytorch yet. It is simpler to deal with a single tensor
        # and since we are only gathering a small amount of data,
        # it should be ok to use all-gather instead of gather.
        rank_name_to_time = torch.zeros(
            (world_size, len(names)), dtype=torch.float, device=torch.cuda.current_device()
        )
        for i, name in enumerate(names):
            if name in self._timers:
                # Here we don't need to pass the barrier flag as all
                # the processes are already in sync. This avoids the
                # issue of different timers having different barrier
                # groups inside their class.
                rank_name_to_time[rank, i] = self._timers[name].elapsed(reset=reset)

        # See the note above for why we are not using gather.
        dist_all_gather_func(rank_name_to_time.view(-1), rank_name_to_time[rank, :].view(-1))

        return rank_name_to_time

    def _get_global_min_max_time(self, names, reset, barrier, normalizer):
        """Report only min and max times across all ranks."""

        rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset, barrier)
        name_to_min_max_time = {}
        for i, name in enumerate(names):
            rank_to_time = rank_name_to_time[:, i]
            # filter out the ones we did not have any timings for
            rank_to_time = rank_to_time[rank_to_time > 0.0]
            # If the timer exists:
            if rank_to_time.numel() > 0:
                name_to_min_max_time[name] = (
                    rank_to_time.min().item() / normalizer,
                    rank_to_time.max().item() / normalizer,
                )
        return name_to_min_max_time

    def _get_global_min_max_time_string(self, names, reset, barrier, normalizer, max_only):
        """Report strings for max/minmax times across all ranks."""
        name_to_min_max_time = self._get_global_min_max_time(names, reset, barrier, normalizer)
        if not name_to_min_max_time:
            return None
        if max_only:
            output_string = 'max time across ranks (ms):'
        else:
            output_string = '(min, max) time across ranks (ms):'
        for name in name_to_min_max_time:
            min_time, max_time = name_to_min_max_time[name]
            if max_only:
                output_string += '\n    {}: {:.2f}'.format((name + ' ').ljust(48, '.'), max_time)
            else:
                output_string += '\n    {}: ({:.2f}, {:.2f})'.format(
                    (name + ' ').ljust(48, '.'), min_time, max_time
                )
        return output_string

    def _get_all_ranks_time_string(self, names, reset, barrier, normalizer):
        """Report times across all ranks."""
        rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset, barrier)

        output_string = 'times across ranks (ms):'
        no_reported_timing = True
        for i, name in enumerate(names):
            not_yet_found = True
            for rank in range(torch.distributed.get_world_size()):
                if rank_name_to_time[rank, i] > 0:
                    no_reported_timing = False
                    if not_yet_found:
                        not_yet_found = False
                        output_string += '\n  {}:'.format(name)
                    output_string += '\n     rank {:2d}: {:.2f}'.format(
                        rank, rank_name_to_time[rank, i] / normalizer
                    )
        if no_reported_timing:
            return None
        return output_string

    def get_all_timers_string(
        self,
        names: List[str] = None,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
    ):
        """Returns the output string with logged timer values according to configured options.

        Args:
            names (List[str]): Names of the timers to log. If None, all registered timers are
                               fetched. Defaults to None.
            normalizer (float, optional): Normalizes the timer values by the factor.
                                          Defaults to 1.0.
            reset (bool, optional): Whether to reset timer values after logging. Defaults to True.
            barrier (bool, optional): Whether to do a global barrier before time measurments.
                                      Defaults to False.

        Raises:
            Exception: Raises if log option is invalid.

        Returns:
            str: Formatted string with the timer values.
        """

        if names == None:  # get all registered timers
            names = self._timers.keys()

        assert normalizer > 0.0
        if self._log_option in ['max', 'minmax']:
            max_only = False
            if self._log_option == 'max':
                max_only = True
            output_string = self._get_global_min_max_time_string(
                names, reset, barrier, normalizer / 1000.0, max_only
            )
        elif self._log_option == 'all':
            output_string = self._get_all_ranks_time_string(
                names, reset, barrier, normalizer / 1000.0
            )
        else:
            raise Exception('unknown timing log option {}'.format(self._log_option))
        return output_string

    def log(
        self,
        names: List[str],
        rank: int = None,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
    ):
        """logs the timers passed in names to stdout. Example usage is to log average per step
           value for timer 'foo', this function can be called with normalizer factor set to logging
           interval.

        Args:
            names (List[str]): Names of the timers to log.
            rank (int, optional): logs the timers to a specific rank. If set to None, logs to the
                                  last rank. Defaults to None.
            normalizer (float, optional): Normalizes the timer values by the factor.
                                          Defaults to 1.0.
            reset (bool, optional): Whether to reset timer values after logging. Defaults to True.
            barrier (bool, optional): Whether to do a global barrier before time measurments.
                                      Defaults to False.
        """

        output_string = self.get_all_timers_string(names, normalizer, reset, barrier)
        # If no input rank is provided, log on last rank.
        if rank is None:
            rank = torch.distributed.get_world_size() - 1
        if rank == torch.distributed.get_rank() and output_string is not None:
            print(output_string, flush=True)

    def log_with_energy(
        self,
        names: List[str],
        rank: int = None,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
    ):
        """logs the timers with energy consumption information.

        Args:
            names (List[str]): Names of the timers to log.
            rank (int, optional): logs the timers to a specific rank. If set to None, logs to the
                                  last rank. Defaults to None.
            normalizer (float, optional): Normalizes the timer values by the factor.
                                          Defaults to 1.0.
            reset (bool, optional): Whether to reset timer values after logging. Defaults to True.
            barrier (bool, optional): Whether to do a global barrier before time measurments.
                                      Defaults to False.
        """
        # Get timing string
        timing_string = self.get_all_timers_string(names, normalizer, reset, barrier)
        
        # Add energy information if available
        energy_string = self._get_energy_string(names)
        
        # Combine output
        if timing_string or energy_string:
            output_parts = []
            if timing_string:
                output_parts.append(timing_string)
            if energy_string:
                output_parts.append(energy_string)
            output_string = '\n'.join(output_parts)
        else:
            output_string = None

        # If no input rank is provided, log on last rank.
        if rank is None:
            rank = torch.distributed.get_world_size() - 1
        if rank == torch.distributed.get_rank() and output_string is not None:
            print(output_string, flush=True)

    def _get_energy_string(self, names: List[str]):
        """Get formatted string with energy consumption information."""
        if not self._enable_energy_monitoring:
            return None
        
        energy_data = []
        for name in names:
            if name in self._timers and isinstance(self._timers[name], Timer):
                timer = self._timers[name]
                if hasattr(timer, 'energy_consumed'):
                    energy = timer.energy_consumed()
                    if energy > 0:
                        energy_data.append((name, energy))
        
        if not energy_data:
            return None
        
        output_string = 'energy consumption (J):'
        for name, energy in energy_data:
            output_string += '\n    {}: {:.3f}'.format((name + ' ').ljust(48, '.'), energy)
        
        return output_string

    def write(
        self,
        names: List[str],
        writer,
        iteration: int,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
    ):
        """Write timers to a tensorboard writer.
        Note that we only report maximum time across ranks to tensorboard.

        Args:
            names (List[str]): Names of the timers to log.
            writer (SummaryWriter): Tensorboard SummaryWriter object
            iteration (int): Current iteration.
            normalizer (float, optional): Normalizes the timer values by the factor.
                                          Defaults to 1.0.
            reset (bool, optional): Whether to reset timer values after logging. Defaults to True.
            barrier (bool, optional): Whether to do a global barrier before time measurments.
                                      Defaults to False.
        """
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        name_to_min_max_time = self._get_global_min_max_time(names, reset, barrier, normalizer)
        if writer is not None:
            for name in name_to_min_max_time:
                _, max_time = name_to_min_max_time[name]
                writer.add_scalar(name + '-time', max_time, iteration)
