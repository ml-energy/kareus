cd mlp
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_mlp.py --use_effective_energy --normalize_objectives > bo_search_mlp_llama3.2_3b.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_mlp_backward.py --use_effective_energy --normalize_objectives > bo_search_mlp_backward_llama3.2_3b.log 2>&1
cd ..

cd qkv_ar
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_attn.py --use_effective_energy --normalize_objectives > bo_search_attn_llama3.2_3b.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_attn_backward.py --use_effective_energy --normalize_objectives > bo_search_attn_backward_llama3.2_3b.log 2>&1
cd ..

cd qkv_ag
CUDA_VISIBLE_DEVICES=0,1 python bo_search_attn.py --use_effective_energy --normalize_objectives > bo_search_attn_llama3.2_3b.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 python bo_search_attn_backward.py --use_effective_energy --normalize_objectives > bo_search_attn_backward_llama3.2_3b.log 2>&1
cd ..

cd mlp
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_mlp.py -b 8 -s 8192 --use_effective_energy --normalize_objectives > bo_search_mlp_llama3.2_3b_8_8192.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_mlp_backward.py -b 8 -s 8192 --use_effective_energy --normalize_objectives > bo_search_mlp_backward_llama3.2_3b_8_8192.log 2>&1
cd ..

cd qkv_ar
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_attn.py -b 8 -s 8192 --use_effective_energy --normalize_objectives > bo_search_attn_llama3.2_3b_8_8192.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_attn_backward.py -b 8 -s 8192 --use_effective_energy --normalize_objectives > bo_search_attn_backward_llama3.2_3b_8_8192.log 2>&1
cd ..

cd qkv_ag
CUDA_VISIBLE_DEVICES=0,1 python bo_search_attn.py -b 8 -s 8192 --use_effective_energy --normalize_objectives > bo_search_attn_llama3.2_3b_8_8192.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 python bo_search_attn_backward.py -b 8 -s 8192 --use_effective_energy --normalize_objectives > bo_search_attn_backward_llama3.2_3b_8_8192.log 2>&1
cd ..

cd attn_oproj
CUDA_VISIBLE_DEVICES=0,1 python bo_search_a_ag.py --use_effective_energy --normalize_objectives > bo_search_a_ag_llama3.2_3b.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 python bo_search_a_rs.py --use_effective_energy --normalize_objectives > bo_search_a_rs_llama3.2_3b.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 python bo_search_ao_ag.py --use_effective_energy --normalize_objectives > bo_search_ao_ag_llama3.2_3b.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_ao_ar.py --use_effective_energy --normalize_objectives > bo_search_ao_ar_llama3.2_3b.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 python bo_search_o_ag.py --use_effective_energy --normalize_objectives > bo_search_o_ag_llama3.2_3b.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_o_ar.py --use_effective_energy --normalize_objectives > bo_search_o_ar_llama3.2_3b.log 2>&1
cd ..

cd attn_oproj
CUDA_VISIBLE_DEVICES=0,1 python bo_search_a_ag.py -b 8 -s 8192 --use_effective_energy --normalize_objectives > bo_search_a_ag_llama3.2_3b_8_8192.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 python bo_search_a_rs.py -b 8 -s 8192 --use_effective_energy --normalize_objectives > bo_search_a_rs_llama3.2_3b_8_8192.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 python bo_search_ao_ag.py -b 8 -s 8192 --use_effective_energy --normalize_objectives > bo_search_ao_ag_llama3.2_3b_8_8192.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_ao_ar.py -b 8 -s 8192 --use_effective_energy --normalize_objectives > bo_search_ao_ar_llama3.2_3b_8_8192.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 python bo_search_o_ag.py -b 8 -s 8192 --use_effective_energy --normalize_objectives > bo_search_o_ag_llama3.2_3b_8_8192.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 python bo_search_o_ar.py -b 8 -s 8192 --use_effective_energy --normalize_objectives > bo_search_o_ar_llama3.2_3b_8_8192.log 2>&1
cd ..