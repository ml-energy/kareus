import zmq
import zmq.asyncio
import socket
from typing import List
from torch.multiprocessing.spawn import spawn
import pickle
import asyncio

from cfuser.logger import init_logger

logger = init_logger(__name__)


def find_free_port():
    """Find a free port on the system.

    Returns:
        int: An available port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def find_free_ports(num_ports):
    """
    Find a list of consecutive free ports

    Args:
        num_ports (int): Number of consecutive ports needed

    Returns:
        list: List of consecutive free port numbers
    """
    import socket

    def is_port_free(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return True
            except OSError:
                return False

    # Start checking from a common range (e.g., 8000)
    start_port = 8000
    max_port = 65535

    while start_port + num_ports <= max_port:
        ports = list(range(start_port, start_port + num_ports))
        if all(is_port_free(port) for port in ports):
            return ports
        start_port += 1

    raise RuntimeError(f"Could not find {num_ports} consecutive free ports")


class Router:
    def __init__(self, address: str, port: int, dealer_num: int):
        self.context = zmq.asyncio.Context()
        self.router = self.context.socket(zmq.ROUTER)
        self.router.bind(f"tcp://{address}:{port}")
        self.dealer_num = dealer_num

        self.workers = {}

    async def init(self):
        # register all workers
        for _ in range(0, self.dealer_num):
            # ZMQ ROUTER socket receives messages in format:
            #   [identity, empty delimiter, payload]

            message_parts = await self.router.recv_multipart()
            identity = message_parts[0]
            worker_id = identity.decode()
            logger.info(f"router recv from worker {worker_id}: {pickle.loads(message_parts[-1])}")

            if worker_id not in self.workers:
                self.workers[worker_id] = identity

        await self.send_to_workers(list(self.workers.keys()), "router init ok!")

    async def send_to_worker(self, worker_id: str, msg: object):
        if worker_id not in self.workers:
            raise ValueError(f"Worker {worker_id} not registered")
        identity = self.workers[worker_id]
        # TODO(Junzhe Ma): remove pickle
        await self.router.send_multipart([identity, b"", pickle.dumps(msg)])

    async def recv_from_worker(self):
        message_parts = await self.router.recv_multipart()
        identity = message_parts[0]
        worker_id = identity.decode()
        # Skip the empty delimiter and return the payload
        return worker_id, pickle.loads(message_parts[-1])

    async def send_to_workers(self, worker_ids: List[str], msg: object):
        # TODO(Jeff Ma): async gather
        encoded_msg = pickle.dumps(msg)
        for worker_id in worker_ids:
            if worker_id not in self.workers:
                raise ValueError(f"Worker {worker_id} not registered")
            identity = self.workers[worker_id]
            await self.router.send_multipart([identity, b"", encoded_msg])


class Dealer:
    def __init__(self, router_address: str, router_port: int, worker_id: str):
        self.context = zmq.Context()
        self.dealer = self.context.socket(zmq.DEALER)
        self.dealer.identity = f"{worker_id}".encode()
        self.dealer.connect(f"tcp://{router_address}:{router_port}")

        # init register to router
        self.send_to_router("init ok")
        # router ok
        res = self.recv_from_router()
        logger.info(f"dealer {worker_id} recv from router: {res}")

    def send_to_router(self, msg: object):
        self.dealer.send_multipart([b"", pickle.dumps(msg)])

    def recv_from_router(self):
        response = self.dealer.recv_multipart()
        # Skip the empty delimiter and return the payload
        return pickle.loads(response[-1])


async def test_dealer_router(rank, world_size, router_address, router_port):
    if rank == 0:
        router = Router(router_address, router_port, world_size - 1)
        await router.init()

        for _ in range(router.dealer_num):
            worker_id, msg = await router.recv_from_worker()
            logger.info(f"router recv from worker {worker_id}: {msg}")
    else:
        identity = f"Worker-{rank}"

        dealer = Dealer(router_address, router_port, identity)

        dealer.send_to_router(f"{identity} pinging router")
        logger.info(f"{identity} ping sent to router")


def run_test_dealer_router(rank, world_size, router_address, router_port):
    asyncio.run(test_dealer_router(rank, world_size, router_address, router_port))


if __name__ == "__main__":
    world_size = 5
    router_port = find_free_port()

    spawn(
        run_test_dealer_router,
        nprocs=world_size,
        args=(world_size, "localhost", router_port),
    )
