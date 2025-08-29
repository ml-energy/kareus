from typing import Dict, Type, Union
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from cfuser.logger import init_logger
from .base_pipeline import cFuserPipelineBaseWrapper

logger = init_logger(__name__)


class cFuserPipelineWrapperRegister:
    _XFUSER_PIPE_MAPPING: Dict[Type[DiffusionPipeline], Type[cFuserPipelineBaseWrapper]] = {}

    @classmethod
    def register(cls, origin_pipe_class: Type[DiffusionPipeline]):
        def decorator(cfuser_pipe_class: Type[cFuserPipelineBaseWrapper]):
            if not issubclass(cfuser_pipe_class, cFuserPipelineBaseWrapper):
                raise ValueError(f"{cfuser_pipe_class} is not a subclass of" f" cFuserPipelineBaseWrapper")
            cls._XFUSER_PIPE_MAPPING[origin_pipe_class] = cfuser_pipe_class
            return cfuser_pipe_class

        return decorator

    @classmethod
    def get_class(cls, pipe: Union[DiffusionPipeline, Type[DiffusionPipeline]]) -> Type[cFuserPipelineBaseWrapper]:
        if isinstance(pipe, type):
            candidate = None
            candidate_origin = None
            for (
                origin_model_class,
                cfuser_model_class,
            ) in cls._XFUSER_PIPE_MAPPING.items():
                if issubclass(pipe, origin_model_class):
                    if (candidate is None and candidate_origin is None) or issubclass(
                        origin_model_class, candidate_origin
                    ):
                        candidate_origin = origin_model_class
                        candidate = cfuser_model_class
            if candidate is None:
                raise ValueError(f"Diffusion Pipeline class {pipe} " f"is not supported by cFuser")
            else:
                return candidate
        elif isinstance(pipe, DiffusionPipeline):
            candidate = None
            candidate_origin = None
            for (
                origin_model_class,
                cfuser_model_class,
            ) in cls._XFUSER_PIPE_MAPPING.items():
                if isinstance(pipe, origin_model_class):
                    if (candidate is None and candidate_origin is None) or issubclass(
                        origin_model_class, candidate_origin
                    ):
                        candidate_origin = origin_model_class
                        candidate = cfuser_model_class

            if candidate is None:
                raise ValueError(f"Diffusion Pipeline class {pipe.__class__} " f"is not supported by cFuser")
            else:
                return candidate
        else:
            raise ValueError(f"Unsupported type {type(pipe)} for pipe")
