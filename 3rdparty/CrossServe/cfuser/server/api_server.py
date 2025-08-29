from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union

from fastapi.responses import Response

from cfuser.engine.runtime import CServeRuntime
from cfuser.config import InputConfig
from cfuser.config.args import ServerArgs
from cfuser.utils import get_gen_req_id
import asyncio

from cfuser.logger import init_logger

logger = init_logger(__name__)

app = FastAPI(title="CServe API Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
cserve_runtime: Optional[CServeRuntime] = None

request_id_lock = asyncio.Lock()  # Ensure thread safety


class RequestInput(BaseModel):
    prompt: Union[str, List[str]]
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 20
    guidance_scale: float = 0.0
    seed: Optional[int] = None
    output_type: str = "pil"
    batch_size: int = 1

    def __post_init__(self):
        # TODO: support batch generation
        assert self.batch_size == 1, "batch_size must be 1"
        assert isinstance(self.prompt, str), "prompt must be a string"


@app.post("/v1/generate")
async def generate(request: RequestInput):
    """
    The api server needs to return the response for this RequestInput.
    Therefore, await cserve_runtime.generate(config)
    must return the output of the same request.
    api server only accepts single RequestInput

    @Runyu Lu
    Propose: rename InputConfig to GenerationRequest,
    bc it **is** a request corresponding some unique response.

    """
    if cserve_runtime is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    async with request_id_lock:
        rid = get_gen_req_id()

    try:
        config = InputConfig(
            prompt=request.prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed,
            output_type=request.output_type,
            batch_size=request.batch_size,
        )

        logger.info(f"Received request {rid}")
        output = await cserve_runtime.generate(config, rid)

        if output.image is None:
            return {"output": None}

        # Convert output to base64 if needed
        if request.output_type == "pil":
            # Convert PIL images to base64
            import base64
            import io

            images = []
            # for img in output.image:
            img = output.image
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            images.append(img_str)

            return {"images": images}

        return {"images": [output.image]}

    except Exception as e:
        import traceback

        error_detail = f"Error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/health")
async def health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)


def create_runtime(server_args: ServerArgs):
    global cserve_runtime
    cserve_runtime = CServeRuntime(server_args=server_args)
    return cserve_runtime
