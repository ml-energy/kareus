export NCCL_MAX_NCHANNELS=8
export NCCL_MIN_NCHANNELS=8
export NCCL_NTHREADS=1024
python test_all_reduce.py