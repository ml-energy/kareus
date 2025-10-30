cd attention
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bo_search_attn.py --use_effective_energy --normalize_objectives > bo_search_attn.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bo_search_attn_backward.py --use_effective_energy --normalize_objectives > bo_search_attn_bwd.log 2>&1
cd ..

cd mlp
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bo_search_mlp.py --use_effective_energy --normalize_objectives > bo_search_mlp.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bo_search_mlp_backward.py --use_effective_energy --normalize_objectives > bo_search_mlp_bwd.log 2>&1
cd ..
