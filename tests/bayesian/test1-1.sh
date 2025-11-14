cd mlp
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_mlp.py --batch_size 8 --seq_len 8192 --use_effective_energy --normalize_objectives > bo_search_mlp_llama3.2_3b_8_8192.log 2>&1
cd ..

cd qkv_ar
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_attn.py --batch_size 8 --seq_len 8192 --use_effective_energy --normalize_objectives > bo_search_attn_llama3.2_3b_8_8192.log 2>&1
cd ..

cd attn_oproj
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_ao_ar.py --batch_size 8 --seq_len 8192 --use_effective_energy --normalize_objectives > bo_search_ao_ar_llama3.2_3b_8_8192.log 2>&1
cd ..

cd mlp
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_mlp.py --batch_size 16 --seq_len 4096 --use_effective_energy --normalize_objectives > bo_search_mlp_llama3.2_3b_8_8192.log 2>&1
cd ..

cd qkv_ar
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_attn.py --batch_size 16 --seq_len 4096 --use_effective_energy --normalize_objectives > bo_search_attn_llama3.2_3b_8_8192.log 2>&1
cd ..

cd attn_oproj
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_ao_ar.py --batch_size 16 --seq_len 4096 --use_effective_energy --normalize_objectives > bo_search_ao_ar_llama3.2_3b_8_8192.log 2>&1
cd ..