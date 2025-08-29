from .config import cFuserArgs
from .model_executor.pipelines import cFuserFluxPipeline

import os

os.environ["NCCL_BUFFSIZE"] = f"{1048576 * 10}"  # 10MB
# still don't know why this is needed
# https://github.com/NVIDIA/nccl/issues/1252
# https://dev-discuss.pytorch.org/t/memcpy-based-p2p-communication-for-pipeline-parallelism-instead-nccl/2184
# print(f"NCCL_BUFFSIZE: {os.environ['NCCL_BUFFSIZE']}")

__all__ = ["cFuserArgs", "cFuserFluxPipeline"]
