import sys
import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, Union, List, Tuple

import torch
import torch.distributed as dist
from torch.distributed.argparse_util import env

from cfuser.config.config import (
    ModelConfig,
    RuntimeConfig,
    SequenceParallelConfig,
    ParallelConfig,
    EngineConfig,
    InputConfig,
)

from cfuser.logger import init_logger

logger = init_logger(__name__)


class FlexibleArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""

    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]

        # Convert underscores to dashes and vice versa in argument names
        processed_args = []
        for arg in args:
            if arg.startswith("--"):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    key = "--" + key[len("--") :].replace("-", "_")
                    processed_args.append(f"{key}={value}")
                else:
                    processed_args.append("--" + arg[len("--") :].replace("-", "_"))
            else:
                processed_args.append(arg)

        return super().parse_args(processed_args, namespace)


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


@dataclass
class cFuserArgs:
    """Arguments for cFuser engine."""

    # Model arguments
    model: str
    download_dir: Optional[str] = None
    trust_remote_code: bool = False

    # Runtime arguments
    use_cuda_graph: bool = True
    use_parallel_vae: bool = False
    use_profiler: bool = False
    use_torch_compile: bool = False
    use_onediff: bool = False

    # Parallel arguments
    # sequence parallel
    ulysses_degree: Optional[int] = None
    ring_degree: Optional[int] = None

    # Input arguments
    height: int = 1024
    width: int = 1024
    num_frames: int = 49
    batch_size: int = 1
    num_inference_steps: int = 20
    max_sequence_length: int = 256
    prompt: Union[str, List[str]] = ""
    negative_prompt: Union[str, List[str]] = ""
    no_use_resolution_binning: bool = False
    seed: int = 42
    output_type: str = "pil"
    enable_sequential_cpu_offload: bool = False

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Shared CLI arguments for cFuser engine."""

        # Model arguments
        model_group = parser.add_argument_group("Model arguments")
        model_group.add_argument(
            "--model",
            type=str,
            default="black-forest-labs/FLUX.1-dev",
            help="Name or path of the huggingface model to use.",
        )
        model_group.add_argument(
            "--download-dir",
            type=nullable_str,
            default=cFuserArgs.download_dir,
            help="Directory to download and load the weights, default to the default cache dir of huggingface.",
        )
        model_group.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Trust remote code from huggingface.",
        )

        # Runtime arguments
        runtime_group = parser.add_argument_group("Runtime arguments")
        runtime_group.add_argument("--use_cuda_graph", action="store_true")
        runtime_group.add_argument("--use_parallel_vae", action="store_true")
        runtime_group.add_argument("--use_profiler", action="store_true")
        runtime_group.add_argument(
            "--use_torch_compile",
            action="store_true",
            help="Enable torch.compile to accelerate inference in a single card",
        )
        runtime_group.add_argument(
            "--use_onediff",
            action="store_true",
            help="Enable onediff to accelerate inference in a single card",
        )

        # Parallel arguments
        parallel_group = parser.add_argument_group("Parallel arguments")
        parallel_group.add_argument(
            "--ulysses_degree",
            type=int,
            default=None,
            help="Ulysses sequence parallel degree. Used in attention layer.",
        )
        parallel_group.add_argument(
            "--ring_degree",
            type=int,
            default=None,
            help="Ring sequence parallel degree. Used in attention layer.",
        )

        # Input arguments
        input_group = parser.add_argument_group("Input Options")
        input_group.add_argument("--batch_size", type=int, default=1)
        input_group.add_argument("--height", type=int, default=1024, help="The height of image")
        input_group.add_argument("--width", type=int, default=1024, help="The width of image")
        input_group.add_argument("--num_frames", type=int, default=49, help="The frames of video")
        input_group.add_argument("--prompt", type=str, nargs="*", default="", help="Prompt for the model.")
        input_group.add_argument("--no_use_resolution_binning", action="store_true")
        input_group.add_argument(
            "--negative_prompt",
            type=str,
            nargs="*",
            default="",
            help="Negative prompt for the model.",
        )
        input_group.add_argument(
            "--num_inference_steps",
            type=int,
            default=20,
            help="Number of inference steps.",
        )
        input_group.add_argument(
            "--max_sequence_length",
            type=int,
            default=256,
            help="Max sequencen length of prompt",
        )

        # Runtime arguments
        runtime_group.add_argument("--seed", type=int, default=42, help="Random seed for operations.")
        runtime_group.add_argument(
            "--output_type",
            type=str,
            default="pil",
            help="Output type of the pipeline.",
        )
        runtime_group.add_argument(
            "--enable_sequential_cpu_offload",
            action="store_true",
            help="Offloading the weights to the CPU.",
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_config(self) -> Tuple[EngineConfig, InputConfig]:

        # if not torch.distributed.is_initialized():
        # logger.warning(
        #     "Distributed environment is not initialized. " "Initializing..."
        # )
        # init_distributed_environment()

        model_config = ModelConfig(
            model=self.model,
            download_dir=self.download_dir,
            trust_remote_code=self.trust_remote_code,
        )

        runtime_config = RuntimeConfig(
            use_cuda_graph=self.use_cuda_graph,
            use_parallel_vae=self.use_parallel_vae,
            use_profiler=self.use_profiler,
            use_torch_compile=self.use_torch_compile,
            use_onediff=self.use_onediff,
        )

        sp_config = SequenceParallelConfig(
            ulysses_degree=self.ulysses_degree,
            ring_degree=self.ring_degree,
        )

        parallel_config = ParallelConfig(
            sp_config=sp_config,
        )

        engine_config = EngineConfig(
            model_config=model_config,
            runtime_config=runtime_config,
            parallel_config=parallel_config,
        )

        if isinstance(self.prompt, list):
            if len(self.prompt) < self.batch_size:
                self.prompt = self.prompt + [""] * (self.batch_size - len(self.prompt))
        elif self.batch_size > 1:
            self.prompt = [self.prompt] * self.batch_size

        input_config = InputConfig(
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            use_resolution_binning=not self.no_use_resolution_binning,
            prompt=self.prompt,
            batch_size=len(self.prompt) if isinstance(self.prompt, list) else 1,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            max_sequence_length=self.max_sequence_length,
            seed=self.seed,
            output_type=self.output_type,
        )

        return engine_config, input_config


@dataclass
class ServerArgs:
    """Arguments for cFuser server."""

    nnodes: int = 1
    nproc_per_node: int = 1
    master_addr: str = "127.0.0.1"
    master_port: int = 1037
    schedule_logic: str = "naive"

    engine_config: EngineConfig = None

    def __post_init__(self):
        assert self.nnodes == 1, "Only support single node server"
        if (
            self.engine_config.parallel_config.sp_config.ulysses_degree
            * self.engine_config.parallel_config.sp_config.ring_degree
            != self.nproc_per_node
        ):
            logger.warning(f"nproc_per_node is not equal to ulysses_degree * ring_degree")
            logger.warning(f"reset ulysses_degree to {self.nproc_per_node}, ring_degree to 1")
            self.engine_config.parallel_config.sp_config.ulysses_degree = self.nproc_per_node
            self.engine_config.parallel_config.sp_config.ring_degree = 1
            self.engine_config.parallel_config.sp_degree = self.nproc_per_node
            self.engine_config.parallel_config.ulysses_degree = self.nproc_per_node
            self.engine_config.parallel_config.ring_degree = 1
            assert (
                self.engine_config.parallel_config.sp_config.ulysses_degree
                * self.engine_config.parallel_config.sp_config.ring_degree
                == self.nproc_per_node
            ), "Failed to reset ulysses_degree and ring_degree"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Add server-specific arguments
        parser.add_argument(
            "--nnodes",
            type=str,
            default="1",
            help="Number of nodes",
        )
        parser.add_argument(
            "--nproc-per-node",
            "--nproc_per_node",
            type=str,
            default="1",
            help="Number of workers per node; supported values: [auto, cpu, gpu, int].",
        )
        parser.add_argument(
            "--master-addr",
            "--master_addr",
            default="127.0.0.1",
            type=str,
            action=env,
            help="Address of the master node (rank 0) that only used for static rendezvous. It should "
            "be either the IP address or the hostname of rank 0. For single node multi-proc training "
            "the --master-addr can simply be 127.0.0.1; IPv6 should have the pattern "
            "`[0:0:0:0:0:0:0:1]`.",
        )
        parser.add_argument(
            "--master-port",
            "--master_port",
            default=1037,
            type=int,
            action=env,
            help="Port on the master node (rank 0) to be used for communication during distributed "
            "training. It is only used for static rendezvous.",
        )
        parser.add_argument(
            "--schedule_logic",
            type=str,
            choices=["naive", "scaling_efficient", "disaggregated_scaling_efficient"],
            default="naive",
            help="Scheduler to use",
        )

        # Add engine-related arguments through cFuserArgs
        parser = cFuserArgs.add_cli_args(parser)

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        engine_args = cFuserArgs.from_cli_args(args)
        engine_config, _ = engine_args.create_config()
        return cls(
            nnodes=int(args.nnodes),
            nproc_per_node=int(args.nproc_per_node),
            master_addr=args.master_addr,
            master_port=int(args.master_port),
            schedule_logic=args.schedule_logic,
            engine_config=engine_config,
        )
