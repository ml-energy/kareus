export MASTER_ADDR=172.31.44.236
export MASTER_PORT=29500

torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" kareus_gpt_pretraining.py
