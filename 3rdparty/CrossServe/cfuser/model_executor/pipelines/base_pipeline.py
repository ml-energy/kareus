from abc import ABCMeta, abstractmethod
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.distributed
import torch.nn as nn

from diffusers import DiffusionPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from distvae.modules.adapters.vae.decoder_adapters import DecoderAdapter
from cfuser.config.config import (
    EngineConfig,
    InputConfig,
)
from cfuser.logger import init_logger
from cfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
    initialize_runtime_state,
    get_sequence_parallel_world_size,
)
from cfuser.model_executor.base_wrapper import cFuserBaseWrapper

from cfuser.envs import PACKAGES_CHECKER

PACKAGES_CHECKER.check_diffusers_version()

from cfuser.model_executor.schedulers import *
from cfuser.model_executor.models.transformers import *

try:
    import os
    from onediff.infer_compiler import compile as od_compile

    HAS_OF = True
    os.environ["NEXFORT_FUSE_TIMESTEP_EMBEDDING"] = "0"
    os.environ["NEXFORT_FX_FORCE_TRITON_SDPA"] = "1"
except:
    HAS_OF = False

logger = init_logger(__name__)


class cFuserPipelineBaseWrapper(cFuserBaseWrapper, metaclass=ABCMeta):

    def __init__(
        self,
        pipeline: DiffusionPipeline,
        engine_config: EngineConfig,
    ):
        self.module: DiffusionPipeline
        self.engine_config: EngineConfig = engine_config
        self._init_runtime_state(pipeline=pipeline, engine_config=engine_config)

        # backbone
        transformer = getattr(pipeline, "transformer", None)
        unet = getattr(pipeline, "unet", None)
        # vae
        vae = getattr(pipeline, "vae", None)
        # scheduler
        scheduler = getattr(pipeline, "scheduler", None)

        if transformer is not None:
            pipeline.transformer = self._convert_transformer_backbone(
                transformer,
                enable_torch_compile=engine_config.runtime_config.use_torch_compile,
                enable_onediff=engine_config.runtime_config.use_onediff,
            )
        elif unet is not None:
            assert False, "No UNet"

        if scheduler is not None:
            pipeline.scheduler = self._convert_scheduler(scheduler)

        if vae is not None and engine_config.runtime_config.use_parallel_vae:
            pipeline.vae = self._convert_vae(vae)

        super().__init__(module=pipeline)

    def reset_activation_cache(self):
        if hasattr(self.module, "transformer") and hasattr(self.module.transformer, "reset_activation_cache"):
            self.module.transformer.reset_activation_cache()
        if hasattr(self.module, "unet") and hasattr(self.module.unet, "reset_activation_cache"):
            self.module.unet.reset_activation_cache()
        if hasattr(self.module, "vae") and hasattr(self.module.vae, "reset_activation_cache"):
            self.module.vae.reset_activation_cache()
        if hasattr(self.module, "scheduler") and hasattr(self.module.scheduler, "reset_activation_cache"):
            self.module.scheduler.reset_activation_cache()

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        return self

    @staticmethod
    def check_to_use_naive_forward(func):
        @wraps(func)
        def check_naive_forward_fn(self, *args, **kwargs):
            if get_sequence_parallel_world_size() == 1:
                return self.module(*args, **kwargs)
            else:
                return func(self, *args, **kwargs)

        return check_naive_forward_fn

    @staticmethod
    def check_model_parallel_state(
        sequence_parallel_available: bool = True,
    ):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not sequence_parallel_available and get_runtime_state().parallel_config.sp_degree > 1:
                    raise RuntimeError("Sequence parallelism is not supported by the model")
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def forward(self):
        pass

    def prepare_run(self, input_config: InputConfig, steps: int = 3, sync_steps: int = 1):
        prompt = [""] * input_config.batch_size if input_config.batch_size > 1 else ""
        warmup_steps = get_runtime_state().runtime_config.warmup_steps
        get_runtime_state().runtime_config.warmup_steps = sync_steps
        self.__call__(
            height=input_config.height,
            width=input_config.width,
            prompt=prompt,
            use_resolution_binning=input_config.use_resolution_binning,
            num_inference_steps=steps,
            output_type="latent",
            generator=torch.Generator(device="cuda").manual_seed(42),
        )
        get_runtime_state().runtime_config.warmup_steps = warmup_steps

    def latte_prepare_run(self, input_config: InputConfig, steps: int = 3, sync_steps: int = 1):
        prompt = [""] * input_config.batch_size if input_config.batch_size > 1 else ""
        warmup_steps = get_runtime_state().runtime_config.warmup_steps
        get_runtime_state().runtime_config.warmup_steps = sync_steps
        self.__call__(
            height=input_config.height,
            width=input_config.width,
            prompt=prompt,
            # use_resolution_binning=input_config.use_resolution_binning,
            num_inference_steps=steps,
            output_type="latent",
            generator=torch.Generator(device="cuda").manual_seed(42),
        )
        get_runtime_state().runtime_config.warmup_steps = warmup_steps

    def _init_runtime_state(self, pipeline: DiffusionPipeline, engine_config: EngineConfig):
        initialize_runtime_state(pipeline=pipeline, engine_config=engine_config)

    def change_runtime_config(self, ulysses_degree=1, ring_degree=1):
        raise NotImplementedError("change_runtime_config is Deprecated")
        assert self.module is not None, "pipeline is not initialized"
        ud = self.engine_config.parallel_config.ulysses_degree
        rd = self.engine_config.parallel_config.ring_degree

        assert (
            ud * rd >= ulysses_degree * ring_degree
        ), "the product of ulysses_degree and ring_degree must be less than or equal to the product of the original ulysses_degree and ring_degree"

        from cfuser.config import ParallelConfig, SequenceParallelConfig

        new_parallel_config = ParallelConfig(
            sp_config=SequenceParallelConfig(
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
            )
        )
        self.engine_config = self.engine_config.change_parallel_config(parallel_config=new_parallel_config)
        logger.info(f"change parallel config from U{ud}xR{rd} to U{ulysses_degree}xR{ring_degree}...")
        initialize_runtime_state(pipeline=self.module, engine_config=self.engine_config)

    def _convert_transformer_backbone(self, transformer: nn.Module, enable_torch_compile: bool, enable_onediff: bool):
        # if get_sequence_parallel_world_size() == 1:
        #     logger.info("Transformer backbone found, but model parallelism is not enabled, " "use naive model")
        # else:
        logger.info("Transformer backbone found, paralleling transformer...")
        wrapper = cFuserTransformerWrappersRegister.get_wrapper(transformer)
        transformer = wrapper(transformer)

        if enable_torch_compile and enable_onediff:
            logger.warning(f"apply --use_torch_compile and --use_onediff togather. we use torch compile only")

        if enable_torch_compile or enable_onediff:
            if getattr(transformer, "forward") is not None:
                if enable_torch_compile:
                    optimized_transformer_forward = torch.compile(getattr(transformer, "forward"))
                elif enable_onediff:
                    # O3: +fp16 reduction
                    if not HAS_OF:
                        raise RuntimeError("install onediff and nexfort to --use_onediff")
                    options = {"mode": "O3"}  # mode can be O2 or O3
                    optimized_transformer_forward = od_compile(
                        getattr(transformer, "forward"),
                        backend="nexfort",
                        options=options,
                    )
                setattr(transformer, "forward", optimized_transformer_forward)
            else:
                raise AttributeError(
                    f"Transformer backbone type: {transformer.__class__.__name__} has no attribute 'forward'"
                )
        return transformer

    def _convert_unet_backbone(
        self,
        unet: nn.Module,
    ):
        logger.info("UNet Backbone found")
        raise NotImplementedError("UNet parallelisation is not supported yet")

    def _convert_scheduler(
        self,
        scheduler: nn.Module,
    ):
        logger.info("Scheduler found, paralleling scheduler...")
        wrapper = cFuserSchedulerWrappersRegister.get_wrapper(scheduler)
        scheduler = wrapper(scheduler)
        return scheduler

    def _convert_vae(
        self,
        vae: AutoencoderKL,
    ):
        logger.info("VAE found, paralleling vae...")
        vae.decoder = DecoderAdapter(vae.decoder)
        return vae

    @abstractmethod
    def __call__(self):
        pass
