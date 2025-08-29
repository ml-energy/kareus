from abc import abstractmethod, ABCMeta
from typing import Dict, List, Optional, Tuple, Type
import torch
import torch.nn as nn

from cfuser.core.distributed import (
    get_sequence_parallel_world_size,
)
from cfuser.core.distributed.runtime_state import get_runtime_state
from cfuser.logger import init_logger
from cfuser.model_executor.models import cFuserModelBaseWrapper

logger = init_logger(__name__)


class cFuserTransformerBaseWrapper(cFuserModelBaseWrapper, metaclass=ABCMeta):
    # transformer: original transformer model (for example Transformer2DModel)
    def __init__(
        self,
        transformer: nn.Module,
        submodule_classes_to_wrap: List[Type] = [],
        submodule_name_to_wrap: List = [],
        submodule_addition_args: Dict = {},
    ):
        transformer = self._convert_transformer_for_parallel(
            transformer,
            submodule_classes_to_wrap=submodule_classes_to_wrap,
            submodule_name_to_wrap=submodule_name_to_wrap,
            submodule_addition_args=submodule_addition_args,
        )
        super().__init__(module=transformer)

    def _convert_transformer_for_parallel(
        self,
        transformer: nn.Module,
        submodule_classes_to_wrap: List[Type] = [],
        submodule_name_to_wrap: List = [],
        submodule_addition_args: Dict = {},
    ) -> nn.Module:
        # if get_sequence_parallel_world_size() == 1:
        #     return transformer
        # else:
        transformer = self._wrap_layers(
            model=transformer,
            submodule_classes_to_wrap=submodule_classes_to_wrap,
            submodule_name_to_wrap=submodule_name_to_wrap,
            submodule_addition_args=submodule_addition_args,
        )
        return transformer

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def _get_patch_height_width(self) -> Tuple[int, int]:
        patch_size = get_runtime_state().backbone_patch_size
        vae_scale_factor = get_runtime_state().vae_scale_factor
        width = get_runtime_state().input_config.width // patch_size // vae_scale_factor
        height = get_runtime_state().input_config.height // patch_size // vae_scale_factor

        return height, width
