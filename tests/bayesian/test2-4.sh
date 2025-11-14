# cd qkv_ag
# CUDA_VISIBLE_DEVICES=6,7 python bo_search_attn_backward.py --batch_size 8 --seq_len 8192 --use_effective_energy --normalize_objectives > bo_search_attn_backward_llama3.2_3b_8_8192.log 2>&1

# CUDA_VISIBLE_DEVICES=6,7 python bo_search_attn.py --batch_size 16 --seq_len 4096 --use_effective_energy --normalize_objectives > bo_search_attn_llama3.2_3b_16_4096.log 2>&1
# CUDA_VISIBLE_DEVICES=6,7 python bo_search_attn_backward.py --batch_size 16 --seq_len 4096 --use_effective_energy --normalize_objectives > bo_search_attn_backward_llama3.2_3b_16_4096.log 2>&1
# cd ..

cd attn_oproj
# CUDA_VISIBLE_DEVICES=6,7 python bo_search_a_ag.py --use_effective_energy --normalize_objectives > bo_search_a_ag_llama3.2_3b.log 2>&1
CUDA_VISIBLE_DEVICES=6,7 python bo_search_a_rs.py --batch_size 16 --seq_len 4096 --use_effective_energy --normalize_objectives > bo_search_a_rs_llama3.2_3b.log 2>&1
CUDA_VISIBLE_DEVICES=6,7 python bo_search_ao_ag.py --batch_size 16 --seq_len 4096 --use_effective_energy --normalize_objectives > bo_search_ao_ag_llama3.2_3b.log 2>&1
CUDA_VISIBLE_DEVICES=6,7 python bo_search_o_ag.py --batch_size 16 --seq_len 4096 --use_effective_energy --normalize_objectives > bo_search_o_ag_llama3.2_3b.log 2>&1
cd ..