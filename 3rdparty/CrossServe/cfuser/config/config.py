import os
import torch
import torch.distributed as dist
from packaging import version
from dataclasses import dataclass, fields

from torch import distributed as dist

from cfuser.logger import init_logger
import cfuser.envs as envs
from cfuser.envs import CUDA_VERSION, TORCH_VERSION, PACKAGES_CHECKER

logger = init_logger(__name__)

from typing import Union, Optional, List

env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]
HAS_FLASH_ATTN = env_info["has_flash_attn"]


def check_packages():
    import diffusers

    if not version.parse(diffusers.__version__) > version.parse("0.30.2"):
        raise RuntimeError(
            "This project requires diffusers version > 0.30.2. Currently, you can not install a correct version of diffusers by pip install."
            "Please install it from source code!"
        )


def check_env():
    # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html
    if CUDA_VERSION < version.parse("11.3"):
        raise RuntimeError("NCCL CUDA Graph support requires CUDA 11.3 or above")
    if TORCH_VERSION < version.parse("2.2.0"):
        # https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
        raise RuntimeError(
            "CUDAGraph with NCCL support requires PyTorch 2.2.0 or above. "
            "If it is not released yet, please install nightly built PyTorch "
            "with `pip3 install --pre torch torchvision torchaudio --index-url "
            "https://download.pytorch.org/whl/nightly/cu121`"
        )


@dataclass
class ModelConfig:
    model: str = "black-forest-labs/FLUX.1-dev"
    download_dir: Optional[str] = None
    trust_remote_code: bool = False


@dataclass
class RuntimeConfig:
    dtype: torch.dtype = torch.float16
    use_cuda_graph: bool = False
    use_parallel_vae: bool = False
    use_profiler: bool = False
    use_torch_compile: bool = False
    use_onediff: bool = False

    def __post_init__(self):
        check_packages()
        if self.use_cuda_graph:
            check_env()


@dataclass
class SequenceParallelConfig:
    ulysses_degree: Optional[int] = None
    ring_degree: Optional[int] = None

    def __post_init__(self):
        if self.ulysses_degree is None:
            self.ulysses_degree = 1
            logger.info(f"Ulysses degree not set, " f"using default value {self.ulysses_degree}")
        if self.ring_degree is None:
            self.ring_degree = 1
            logger.info(f"Ring degree not set, " f"using default value {self.ring_degree}")
        self.sp_degree = self.ulysses_degree * self.ring_degree

        if not HAS_FLASH_ATTN and self.ring_degree > 1:
            raise ValueError(f"Flash attention not found. Ring attention not available. Please set ring_degree to 1")


@dataclass
class ParallelConfig:
    sp_config: SequenceParallelConfig

    def __post_init__(self):
        assert self.sp_config is not None, "sp_config must be set"
        # parallel_world_size = self.sp_config.sp_degree
        # world_size = dist.get_world_size()
        # assert parallel_world_size == world_size, (
        #     f"parallel_world_size {parallel_world_size} " f"must be equal to world_size {dist.get_world_size()}"
        # )
        # assert world_size % self.sp_config.sp_degree == 0, "world_size must be divisible by sp_degree"
        self.sp_degree = self.sp_config.sp_degree

        self.ulysses_degree = self.sp_config.ulysses_degree
        self.ring_degree = self.sp_config.ring_degree


@dataclass(frozen=True)
class EngineConfig:
    model_config: ModelConfig
    runtime_config: RuntimeConfig
    parallel_config: ParallelConfig

    def to_dict(self):
        """Return the configs as a dictionary, for use in **kwargs."""
        return dict((field.name, getattr(self, field.name)) for field in fields(self))


@dataclass
# should be rename to GenerationRequest
class InputConfig:
    height: int = 1024
    width: int = 1024
    num_frames: int = 49
    use_resolution_binning: bool = True
    batch_size: int = 1
    prompt: Union[str, List[str]] = ""
    negative_prompt: Union[str, List[str]] = ""
    num_inference_steps: int = 20
    max_sequence_length: int = 256
    seed: Optional[int] = 42
    output_type: str = "pil"
    latency_threshold: Optional[float] = 120
    rid: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.prompt, list):
            assert (
                len(self.prompt) == len(self.negative_prompt) or len(self.negative_prompt) == 0
            ), "prompts and negative_prompts must have the same quantities"
            self.batch_size = self.batch_size or len(self.prompt)
        else:
            self.batch_size = self.batch_size or 1
        assert self.output_type in [
            "pil",
            "latent",
            "pt",
            "pil_latent",
        ], "output_type must be 'pil', 'pt', 'latent' or 'pil_latent'"
