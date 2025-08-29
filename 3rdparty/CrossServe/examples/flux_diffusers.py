import logging
import socket
from datetime import datetime

import torch

from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda:1")
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power


def single_run(batch_size=1, height=1024, width=1024, num_inference_steps=50):
    prompt = "A cat holding a sign that says hello world"
    image = pipe(
        prompt=[prompt] * batch_size,
        height=height,
        width=width,
        guidance_scale=3.5,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    image.save("flux-dev.png")


# code below refer to https://pytorch.org/blog/understanding-gpu-memory-1/

logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000


def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)


def stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)


def export_memory_snapshot(file_prefix: str = "") -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not exporting memory snapshot")
        return

    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{file_prefix}_{host_name}_{timestamp}"

    try:
        logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
        return


def run_flux_with_memory_profiling(batch_size=1, height=1024, width=1024, num_inference_steps=5):
    # Start recording memory snapshot history
    start_record_memory_history()

    # Run Flux
    single_run(
        batch_size=batch_size,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
    )

    # Create the memory snapshot file
    export_memory_snapshot(f"bs{batch_size}-h{height}-w{width}")

    # Stop recording memory snapshot history
    stop_record_memory_history()


# warmup
def warmup(times=3):
    for _ in range(times):
        single_run()


if __name__ == "__main__":
    warmup(5)
    # Run Flux with memory profiling
    run_flux_with_memory_profiling(batch_size=4, height=2048, width=2048, num_inference_steps=5)
