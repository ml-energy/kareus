export MASTER_ADDR=172.31.44.174
export MASTER_PORT=29500
export NCCL_DEBUG=INFO

torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" kareus_gpt_pretraining.py
