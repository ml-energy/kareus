from typing import Dict, Type
import torch
import torch.nn as nn

from cfuser.logger import init_logger
from cfuser.model_executor.layers.base_layer import cFuserLayerBaseWrapper

logger = init_logger(__name__)


class cFuserLayerWrappersRegister:
    _XFUSER_LAYER_MAPPING: Dict[Type[nn.Module], Type[cFuserLayerBaseWrapper]] = {}

    @classmethod
    def register(cls, origin_layer_class: Type[nn.Module]):
        def decorator(cfuser_layer_wrapper: Type[cFuserLayerBaseWrapper]):
            if not issubclass(cfuser_layer_wrapper, cFuserLayerBaseWrapper):
                raise ValueError(
                    f"{cfuser_layer_wrapper.__class__.__name__} is not a " f"subclass of cFuserLayerBaseWrapper"
                )
            cls._XFUSER_LAYER_MAPPING[origin_layer_class] = cfuser_layer_wrapper
            return cfuser_layer_wrapper

        return decorator

    @classmethod
    def get_wrapper(cls, layer: nn.Module) -> cFuserLayerBaseWrapper:
        candidate = None
        candidate_origin = None
        for (
            origin_layer_class,
            cfuser_layer_wrapper,
        ) in cls._XFUSER_LAYER_MAPPING.items():
            if isinstance(layer, origin_layer_class):
                if (
                    (candidate is None and candidate_origin is None)
                    or origin_layer_class == layer.__class__
                    or issubclass(origin_layer_class, candidate_origin)
                ):
                    candidate_origin = origin_layer_class
                    candidate = cfuser_layer_wrapper

        if candidate is None:
            raise ValueError(f"Layer class {layer.__class__.__name__} " f"is not supported by cFuser")
        else:
            return candidate
