import torch
from lightning.pytorch.callbacks import Callback
from typing import Any, Optional, List

from nemo.utils.nvtx import nvtx_range_push, nvtx_range_pop
from nemo.utils.app_state import AppState


class NVTXIterationCallback(Callback):
    """
    A PyTorch Lightning callback that adds NVTX annotations for each training iteration.
    
    This callback provides detailed NVTX profiling markers for different phases of each 
    training iteration, making it easier to analyze performance in NVIDIA Nsight Systems.
    
    Args:
        enable_nvtx (bool): Whether to enable NVTX annotations. If False, the callback 
                           does nothing (useful for conditional enabling).
        detailed_annotations (bool): Whether to add detailed annotations for sub-phases
                                   like data loading, forward pass, etc.
        prefix (str): Prefix for all NVTX range names (default: "training").
    
    Example:
        ```python
        # In your Lightning trainer setup
        nvtx_callback = NVTXIterationCallback(
            enable_nvtx=True,
            detailed_annotations=True,
            prefix="gpt_training"
        )
        trainer = Trainer(callbacks=[nvtx_callback])
        ```
    
    NVTX ranges created:
        - {prefix}.iteration_{step}: Overall iteration
        - {prefix}.iteration_{step}.data_loading: Data loading phase (if detailed)
        - {prefix}.iteration_{step}.forward: Forward pass (if detailed)
        - {prefix}.iteration_{step}.backward: Backward pass (if detailed)
    """
    
    def __init__(
        self, 
        enable_nvtx: bool = True,
        detailed_annotations: bool = False,
        prefix: str = "training"
    ):
        self.enable_nvtx = enable_nvtx
        self.detailed_annotations = detailed_annotations 
        self.prefix = prefix
        self.current_step = None
        self.active_ranges: List[str] = []  # Track active ranges for proper cleanup
        
        # Enable NVTX ranges in AppState if requested
        if self.enable_nvtx:
            app_state = AppState()
            app_state._nvtx_ranges = True

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        """Called when the train batch begins."""
        if not self.enable_nvtx:
            return
            
        global_step = trainer.global_step
        self.current_step = global_step
        
        # Start the main iteration range
        iteration_range = f"{self.prefix}.iteration_{global_step}"
        nvtx_range_push(iteration_range)
        self.active_ranges.append(iteration_range)
        
        if self.detailed_annotations:
            # Start data loading range (this represents the data prep phase)
            data_range = f"{self.prefix}.iteration_{global_step}.data_loading"
            nvtx_range_push(data_range)
            self.active_ranges.append(data_range)

    # def on_before_forward(self, trainer, pl_module, batch, batch_idx: int) -> None:
    #     """Called right before the forward pass."""
    #     if not self.enable_nvtx or not self.detailed_annotations or self.current_step is None:
    #         return
            
    #     # End data loading and start forward pass
    #     # if self.active_ranges and "data_loading" in self.active_ranges[-1]:
    #     if self.active_ranges:  
    #         data_range = self.active_ranges.pop()
    #         # nvtx_range_pop(data_range)
    #         nvtx_range_pop()
            
    #     forward_range = f"{self.prefix}.iteration_{self.current_step}.forward"
    #     nvtx_range_push(forward_range)
    #     self.active_ranges.append(forward_range)

    # def on_after_forward(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
    #     """Called right after the forward pass."""
    #     if not self.enable_nvtx or not self.detailed_annotations or self.current_step is None:
    #         return
            
    #     # End forward pass
    #     # if self.active_ranges and "forward" in self.active_ranges[-1]:
    #     if self.active_ranges:
    #         forward_range = self.active_ranges.pop()
    #         # nvtx_range_pop(forward_range)
    #         nvtx_range_pop()

    # def on_before_backward(self, trainer, pl_module, loss) -> None:
    #     """Called right before the backward pass."""
    #     if not self.enable_nvtx or not self.detailed_annotations or self.current_step is None:
    #         return
            
    #     backward_range = f"{self.prefix}.iteration_{self.current_step}.backward"
    #     nvtx_range_push(backward_range)
    #     self.active_ranges.append(backward_range)

    # def on_after_backward(self, trainer, pl_module) -> None:
    #     """Called right after the backward pass."""
    #     if not self.enable_nvtx or not self.detailed_annotations or self.current_step is None:
    #         return
            
    #     # End backward pass
    #     # if self.active_ranges and "backward" in self.active_ranges[-1]:
    #     if self.active_ranges:
    #         backward_range = self.active_ranges.pop()
    #         # nvtx_range_pop(backward_range)
    #         nvtx_range_pop()

    # def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_closure=None) -> None:
    #     """Called before the optimizer step."""
    #     if not self.enable_nvtx or not self.detailed_annotations:
    #         return
            
    #     # Pop the forward range and start backward range
    #     if self.active_ranges:
    #         nvtx_range_pop()  # End forward range
    #         self.active_ranges.pop()
            
    #         # Start backward range
    #         backward_range = f"{self.prefix}.iteration_{self.current_step}.optimizer"
    #         nvtx_range_push(backward_range)
    #         self.active_ranges.append(backward_range)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        """Called when the train batch ends."""
        if not self.enable_nvtx or self.current_step is None:
            return
            
        # Pop all active ranges in reverse order (LIFO)
        while self.active_ranges:
            range_name = self.active_ranges.pop()
            # nvtx_range_pop(range_name)
            nvtx_range_pop()

        self.current_step = None

    # def on_train_epoch_end(self, trainer, pl_module) -> None:
    #     """Clean up any remaining ranges at epoch end."""
    #     if not self.enable_nvtx:
    #         return
            
    #     # Safety cleanup - pop any remaining ranges
    #     while self.active_ranges:
    #         range_name = self.active_ranges.pop()
    #         try:
    #             nvtx_range_pop(range_name)
    #         except:
    #             pass  # Ignore errors during cleanup