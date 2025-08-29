import os
from cfuser.config import EngineConfig
from cfuser.engine.runner import CServeRunner
from cfuser.core.distributed.parallel_state import init_distributed_environment

from cfuser.logger import init_logger

logger = init_logger(__name__)


def launch_worker(rank, parallel_degree, zmq_router_port, master_addr, master_port, engine_config: EngineConfig):
    # support single node only now
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(parallel_degree)
    os.environ["OMP_NUM_THREADS"] = "1"
    dist_init_method = f"tcp://{master_addr}:{master_port}"
    init_distributed_environment(
        local_rank=rank, rank=rank, world_size=parallel_degree, distributed_init_method=dist_init_method
    )
    runner = CServeRunner(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        zmq_router_port=zmq_router_port,
    )

    runner.serve()
