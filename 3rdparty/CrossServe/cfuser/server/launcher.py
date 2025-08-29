import uvicorn
import asyncio
from cfuser.server.api_server import create_runtime, app
from cfuser.config.args import ServerArgs
from argparse import ArgumentParser


async def main():
    parser = ArgumentParser(description="CServe API Server")
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)

    # Initialize the runtime
    cserve_runtime = create_runtime(server_args)

    await cserve_runtime.init_zmq_router()

    # asyncio.create_task(cserve_runtime.event_loop())

    # Start the server
    config = uvicorn.Config(
        app,
        host=server_args.master_addr,
        port=server_args.master_port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    # Run the server; this will block until the server is stopped
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
