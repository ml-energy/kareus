cd mlp
CUDA_VISIBLE_DEVICES=4,5,6,7 python bo_search_mlp_backward.py --batch_size 8 --seq_len 8192 --use_effective_energy --normalize_objectives > bo_search_mlp_backward_llama3.2_3b_16_4096.log 2>&1
cd ..

cd qkv_ar
CUDA_VISIBLE_DEVICES=4,5,6,7 python bo_search_attn_backward.py --batch_size 8 --seq_len 8192 --use_effective_energy --normalize_objectives > bo_search_attn_backward_llama3.2_3b.log 2>&1
cd ..

cd attn_oproj
CUDA_VISIBLE_DEVICES=4,5,6,7 python bo_search_o_ar.py --batch_size 8 --seq_len 8192 --use_effective_energy --normalize_objectives > bo_search_o_ar_llama3.2_3b.log 2>&1
cd ..

cd mlp
CUDA_VISIBLE_DEVICES=4,5,6,7 python bo_search_mlp_backward.py --batch_size 16 --seq_len 4096 --use_effective_energy --normalize_objectives > bo_search_mlp_backward_llama3.2_3b_16_4096.log 2>&1
cd ..

cd qkv_ar
CUDA_VISIBLE_DEVICES=4,5,6,7 python bo_search_attn_backward.py --batch_size 16 --seq_len 4096 --use_effective_energy --normalize_objectives > bo_search_attn_backward_llama3.2_3b.log 2>&1
cd ..

cd attn_oproj
CUDA_VISIBLE_DEVICES=4,5,6,7 python bo_search_o_ar.py --batch_size 16 --seq_len 4096 --use_effective_energy --normalize_objectives > bo_search_o_ar_llama3.2_3b.log 2>&1
cd ..