cd attention
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bo_search_attn.py --use_effective_energy --normalize_objectives > bo_search_attn.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bo_search_attn_backward.py --use_effective_energy --normalize_objectives > bo_search_attn_bwd.log 2>&1
cd ..

cd mlp
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bo_search_mlp.py --use_effective_energy --normalize_objectives > bo_search_mlp.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bo_search_mlp_backward.py --use_effective_energy --normalize_objectives > bo_search_mlp_bwd.log 2>&1
cd ..

nvidia-smi -i 0,1,2,3,4,5,6,7 --reset-gpu-clocks

# cd attention
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bo_search_attn.py --batch_size 8 --seq_len 8192 --use_effective_energy --normalize_objectives > bo_search_attn_llama3.2_3b_8_8192.log 2>&1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bo_search_attn_backward.py --batch_size 8 --seq_len 8192 --use_effective_energy --normalize_objectives > bo_search_attn_backward_llama3.2_3b_8_8192.log 2>&1
# cd ..

# cd mlp
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bo_search_mlp.py --batch_size 8 --seq_len 8192 --use_effective_energy --normalize_objectives > bo_search_mlp_llama3.2_3b_8_8192.log 2>&1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bo_search_mlp_backward.py --batch_size 8 --seq_len 8192 --use_effective_energy --normalize_objectives > bo_search_mlp_backward_llama3.2_3b_8_8192.log 2>&1
# cd ..

# cd attention
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bo_search_attn.py --batch_size 16 --seq_len 4096 --use_effective_energy --normalize_objectives > bo_search_attn_llama3.2_3b_16_4096.log 2>&1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bo_search_attn_backward.py --batch_size 16 --seq_len 4096 --use_effective_energy --normalize_objectives > bo_search_attn_backward_llama3.2_3b_16_4096.log 2>&1
# cd ..

# cd mlp
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bo_search_mlp.py --batch_size 16 --seq_len 4096 --use_effective_energy --normalize_objectives > bo_search_mlp_llama3.2_3b_16_4096.log 2>&1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bo_search_mlp_backward.py --batch_size 16 --seq_len 4096 --use_effective_energy --normalize_objectives > bo_search_mlp_backward_llama3.2_3b_16_4096.log 2>&1
# cd ..