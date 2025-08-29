from typing import List, Union
from dataclasses import dataclass
import asyncio
import PIL
import numpy as np


@dataclass
class RunnerOutput:
    req_ids: List[int]
    images: Union[List[PIL.Image.Image], np.ndarray]


@dataclass
class EngineOutput:
    req_id: int
    image: Union[PIL.Image.Image, np.ndarray]


@dataclass
class ReqState:
    "class to store the state of a request"
    event: asyncio.Event
    output: EngineOutput | None
    success: bool
