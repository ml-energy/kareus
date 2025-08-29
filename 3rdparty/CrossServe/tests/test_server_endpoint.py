import unittest
import requests
from cfuser.testing import popen_launch_server, kill_process_tree

"""
- Run all tests:
python3 tests/test_server_endpoint.py
- Run a specific test:
python3 -m unittest tests.test_server_endpoint.TestServerEndPoint.test_u4_r1_latent
"""


class TestServerEndPoint(unittest.TestCase):
    model = "black-forest-labs/FLUX.1-dev"
    host = "127.0.0.1"
    port = 1037

    def test_u4_r1_latent(self):
        nnodes = 1
        nproc_per_node = 4
        output_type = "latent"
        ulysses_degree = 4
        ring_degree = 1

        process = popen_launch_server(
            self.model,
            nnodes,
            nproc_per_node,
            self.host,
            self.port,
            output_type,
            ulysses_degree,
            ring_degree,
        )

        base_url = f"http://{self.host}:{self.port}"

        response = requests.post(
            base_url + "/v1/generate",
            json={
                "prompt": "A beautiful sunset over mountains",
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 3,
            },
        )

        _ = response.json()["images"]

        kill_process_tree(process.pid)

    def test_u1_r4_latent(self):
        nnodes = 1
        nproc_per_node = 4
        output_type = "latent"
        ulysses_degree = 1
        ring_degree = 4

        process = popen_launch_server(
            self.model,
            nnodes,
            nproc_per_node,
            self.host,
            self.port,
            output_type,
            ulysses_degree,
            ring_degree,
        )

        base_url = f"http://{self.host}:{self.port}"

        response = requests.post(
            base_url + "/v1/generate",
            json={
                "prompt": "A beautiful sunset over mountains",
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 3,
            },
        )

        _ = response.json()["images"]

        kill_process_tree(process.pid)

    def test_u2_r2_latent(self):
        nnodes = 1
        nproc_per_node = 4
        output_type = "latent"
        ulysses_degree = 2
        ring_degree = 2

        process = popen_launch_server(
            self.model,
            nnodes,
            nproc_per_node,
            self.host,
            self.port,
            output_type,
            ulysses_degree,
            ring_degree,
        )

        base_url = f"http://{self.host}:{self.port}"

        response = requests.post(
            base_url + "/v1/generate",
            json={
                "prompt": "A beautiful sunset over mountains",
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 3,
            },
        )

        _ = response.json()["images"]

        kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
