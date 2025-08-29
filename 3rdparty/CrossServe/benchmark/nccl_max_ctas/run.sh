rm -rf log/benchmark/nccl_max_ctas/nccl_max_ctas.json
for i in {1..32}
do
    export NCCL_MAX_CTAS=$i
    export NCCL_MIN_CTAS=$i
    CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark/nccl_max_ctas/test_nccl_SM.py --bs 8 --seq_len 4096 --world_size 4 --logging
done
