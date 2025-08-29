from typing import Dict, Type
import torch
import torch.nn as nn

from cfuser.logger import init_logger
from cfuser.model_executor.models.transformers.base_transformer import (
    cFuserTransformerBaseWrapper,
)

logger = init_logger(__name__)


class cFuserTransformerWrappersRegister:
    _XFUSER_TRANSFORMER_MAPPING: Dict[Type[nn.Module], Type[cFuserTransformerBaseWrapper]] = {}

    @classmethod
    def register(cls, origin_transformer_class: Type[nn.Module]):
        def decorator(cfuser_transformer_class: Type[nn.Module]):
            if not issubclass(cfuser_transformer_class, cFuserTransformerBaseWrapper):
                raise ValueError(
                    f"{cfuser_transformer_class.__class__.__name__} is not "
                    f"a subclass of cFuserTransformerBaseWrapper"
                )
            cls._XFUSER_TRANSFORMER_MAPPING[origin_transformer_class] = cfuser_transformer_class
            return cfuser_transformer_class

        return decorator

    @classmethod
    def get_wrapper(cls, transformer: nn.Module) -> cFuserTransformerBaseWrapper:
        candidate = None
        candidate_origin = None
        for (
            origin_transformer_class,
            wrapper_class,
        ) in cls._XFUSER_TRANSFORMER_MAPPING.items():
            if isinstance(transformer, origin_transformer_class):
                if (
                    candidate is None
                    or origin_transformer_class == transformer.__class__
                    or issubclass(origin_transformer_class, candidate_origin)
                ):
                    candidate_origin = origin_transformer_class
                    candidate = wrapper_class

        if candidate is None:
            raise ValueError(f"Transformer class {transformer.__class__.__name__} " f"is not supported by cFuser")
        else:
            return candidate
