# CrossServe

> A low-latency and high-throughput Cross-Request Diffusion Model Serving Framework.

## Docker

```bash
docker build -t cserve .

docker run --gpus all --ipc=host \
  -it \
  --rm \
  -v $(pwd):/workspaces/CrossServe \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  cserve
```

## Install
we recommend using docker to install CrossServe.
```bash
git clone git@github.com:cserve-project/CrossServe.git --recursive
pip install -e . # check setup.py
```

> After pip install -e ., hope you could find the `msccl_comm_xxx.so` in `cfuser/msccl_comm/` and `libcustom_nccl_all2all.so` in `cfuser/core/distributed`

## Usage

> firstly you need to use huggingface-cli login to verify your account.
```bash
huggingface-cli login
```

offline inference

```bash
python3 examples/offline_inference.py --nnodes 1 --nproc_per_node 4 \
--master_addr 127.0.0.1 --master_port 1037 \
--output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--ulysses_degree 4 --ring_degree 1 2>&1 | tee log.log
```

## API Server

1. run the server

```bash
# with default FCFS scheduler
python -m cfuser.server.launcher \
--nnodes 1 --nproc_per_node 8 \
--master_addr 127.0.0.1 --master_port 1037 \
--schedule_logic "naive" \
--output_type 'latent' --model "black-forest-labs/FLUX.1-dev" 2>&1 | tee log.log

# with scaling efficient scheduler
python -m cfuser.server.launcher \
--nnodes 1 --nproc_per_node 8 \
--master_addr 127.0.0.1 --master_port 1037 \
--schedule_logic "scaling_efficient" \
--output_type 'latent' --model "black-forest-labs/FLUX.1-dev" 2>&1 | tee log.log

# with disaggregated scaling efficient scheduler
python -m cfuser.server.launcher \
--nnodes 1 --nproc_per_node 8 \
--master_addr 127.0.0.1 --master_port 1037 \
--schedule_logic "disaggregated_scaling_efficient" \
--output_type 'latent' --model "black-forest-labs/FLUX.1-dev" 2>&1 | tee log.log
```
> please run `benchmark/component_scaling_efficiency/run.sh` to get the performance model

2. Make API Calls

> use python to make api calls
```python
import requests

response = requests.post("http://localhost:1037/v1/generate", json={
    "prompt": "A beautiful sunset over mountains",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 20
})

# If output_type is "pil", you'll get base64 encoded images
images = response.json()["images"]
```

> use curl to make api calls
```bash
# with minimal parameters
curl -X POST "http://localhost:1037/v1/generate" -H "Content-Type: application/json" -d '{"prompt": "A beautiful sunset over mountains", "height": 512, "width": 512, "num_inference_steps": 10, "output_type": "latent"}'
```

More details can be found in [doc/api_server.md](doc/api_server.md)

## Benchmark

1. Scaling Efficiency and Batching Benefits

```bash
python benchmark/batching_scaling_benefits/launcher.py --config benchmark/batching_scaling_benefits/config.yml
python benchmark/batching_scaling_benefits/launcher.py --config benchmark/batching_scaling_benefits/config.yml --compile
```

2. Server throughput and latency

```python
# you need to launch the server first, see API Server

# Optionally, create a shape distribution
python3 benchmark/serving/benchmark_serving.py -s benchmark/serving/example_shape_distribution.csv
python3 benchmark/serving/benchmark_serving.py --height 1024 --width 1024 --rate 2 --cv 3 --duration 30 --batch-size 2
python3 benchmark/serving/benchmark_serving.py --height 1024 --width 1024 --num-inference-steps 20
```

## Acknowledgements

> Inference Code Structure is based on [xDiT 0.3.2](https://github.com/xdit-project/xDiT)
> Async Server, Benchmark Code refers to [MuxServe](https://github.com/hao-ai-lab/MuxServe), [vllm](https://github.com/vllm-project/vllm/tree/32b6816e556f69f1672085a6267e8516bcb8e622) and [SGLang](https://github.com/sgl-project/sglang).
> Communication Library refers to [Nanoflow](https://github.com/efeslab/Nanoflow/tree/main) and [Mscclpp](https://github.com/microsoft/mscclpp).
> Ring Attention and Ulysses Attention Implementation refers to [yunchang](https://github.com/feifeibear/long-context-attention) and [ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention).

We thanks the following projects for their great work:
- [xDiT 0.3.2](https://github.com/xdit-project/xDiT)
- [Nanoflow](https://github.com/efeslab/Nanoflow/tree/main)
- [Mscclpp](https://github.com/microsoft/mscclpp)
- [MuxServe](https://github.com/hao-ai-lab/MuxServe)
- [SGLang](https://github.com/sgl-project/sglang)
- [vllm](https://github.com/vllm-project/vllm/tree/32b6816e556f69f1672085a6267e8516bcb8e622)
- [ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention)
- [yunchang](https://github.com/feifeibear/long-context-attention)
