from abc import ABCMeta
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from diffusers import DiffusionPipeline, CogVideoXPipeline
import torch.distributed

from cfuser.config.config import (
    ParallelConfig,
    RuntimeConfig,
    InputConfig,
    EngineConfig,
)
from cfuser.logger import init_logger
from .parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

logger = init_logger(__name__)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RuntimeState(metaclass=ABCMeta):
    parallel_config: ParallelConfig
    runtime_config: RuntimeConfig
    input_config: InputConfig
    ready: bool = False

    def __init__(self, config: EngineConfig):
        self.parallel_config = config.parallel_config
        self.runtime_config = config.runtime_config
        self.input_config = InputConfig()
        self._check_distributed_env(config.parallel_config)

        self.ready = False

    def is_ready(self):
        return self.ready

    def _check_distributed_env(
        self,
        parallel_config: ParallelConfig,
    ):
        if not model_parallel_is_initialized():
            logger.warning("Model parallel is not initialized, initializing...")
            if not torch.distributed.is_initialized():
                init_distributed_environment()
            initialize_model_parallel(
                sequence_parallel_degree=parallel_config.sp_degree,
                ulysses_degree=parallel_config.ulysses_degree,
                ring_degree=parallel_config.ring_degree,
            )
        else:
            logger.info("Model parallel is already initialized")
            if not torch.distributed.is_initialized():
                raise RuntimeError("Distributed environment is not initialized")

    def destory_distributed_env(self):
        if model_parallel_is_initialized():
            destroy_model_parallel()
        destroy_distributed_environment()


class DiTRuntimeState(RuntimeState):
    patch_mode: bool
    vae_scale_factor: int
    vae_scale_factor_spatial: int
    vae_scale_factor_temporal: int
    backbone_patch_size: int

    def __init__(self, pipeline: DiffusionPipeline, config: EngineConfig):
        super().__init__(config)
        self.patch_mode = False

        self._check_model_and_parallel_config(pipeline=pipeline, parallel_config=config.parallel_config)
        self.cogvideox = False

        self._set_model_parameters(
            vae_scale_factor=pipeline.vae_scale_factor,
            backbone_patch_size=pipeline.transformer.config.patch_size,
            backbone_in_channel=pipeline.transformer.config.in_channels,
            backbone_inner_dim=pipeline.transformer.config.num_attention_heads
            * pipeline.transformer.config.attention_head_dim,
        )

    def set_input_parameters(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.input_config.num_inference_steps = num_inference_steps or self.input_config.num_inference_steps

        if seed is not None and seed != self.input_config.seed:
            self.input_config.seed = seed
            set_random_seed(seed)
        if (
            not self.ready
            or (height and self.input_config.height != height)
            or (width and self.input_config.width != width)
            or (batch_size and self.input_config.batch_size != batch_size)
            or not self.ready
        ):
            self._input_size_change(height, width, batch_size)

        self.ready = True

    def _check_model_and_parallel_config(
        self,
        pipeline: DiffusionPipeline,
        parallel_config: ParallelConfig,
    ):
        num_heads = pipeline.transformer.config.num_attention_heads
        ulysses_degree = parallel_config.sp_config.ulysses_degree
        if num_heads % ulysses_degree != 0 or num_heads < ulysses_degree:
            raise RuntimeError(
                f"transformer backbone has {num_heads} heads, which is not "
                f"divisible by or smaller than ulysses_degree "
                f"{ulysses_degree}."
            )

    def _set_model_parameters(
        self,
        vae_scale_factor: int,
        backbone_patch_size: int,
        backbone_inner_dim: int,
        backbone_in_channel: int,
    ):
        self.vae_scale_factor = vae_scale_factor
        self.backbone_patch_size = backbone_patch_size
        self.backbone_inner_dim = backbone_inner_dim
        self.backbone_in_channel = backbone_in_channel

    def _input_size_change(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.input_config.height = height or self.input_config.height
        self.input_config.width = width or self.input_config.width
        self.input_config.batch_size = batch_size or self.input_config.batch_size
        self._calc_patches_metadata()

    def _calc_patches_metadata(self):
        num_sp_patches = get_sequence_parallel_world_size()
        sp_patch_idx = get_sequence_parallel_rank()

        vae_scale_factor = self.vae_scale_factor

        latent_height = self.input_config.height // vae_scale_factor
        latent_width = self.input_config.width // vae_scale_factor

        self.latent_idx = (
            (latent_height // num_sp_patches) * sp_patch_idx * latent_width,
            (latent_height // num_sp_patches) * (sp_patch_idx + 1) * latent_width,
        )

    def _video_input_size_change(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.input_config.height = height or self.input_config.height
        self.input_config.width = width or self.input_config.width
        self.input_config.num_frames = num_frames or self.input_config.num_frames
        self.input_config.batch_size = batch_size or self.input_config.batch_size


_RUNTIMEs: Optional[List[DiTRuntimeState]] = None


def runtime_state_is_initialized():
    return _RUNTIMEs is not None


def get_runtime_state(index_req: int = 0):
    assert _RUNTIMEs is not None, "Runtime state has not been initialized."
    return _RUNTIMEs[index_req]


MAX_RUNTIME_STATES = 20


def initialize_runtime_state(pipeline: DiffusionPipeline, engine_config: EngineConfig):
    global _RUNTIMEs
    if _RUNTIMEs is not None:
        logger.warning("Runtime state is already initialized, reinitializing with pipeline...")
    if hasattr(pipeline, "transformer"):
        _RUNTIMEs = [DiTRuntimeState(pipeline=pipeline, config=engine_config)] * MAX_RUNTIME_STATES
