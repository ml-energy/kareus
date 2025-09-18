cd mlp
CUDA_VISIBLE_DEVICES=2,3 python bo_search_mlp.py --use_effective_energy --normalize_objectives > bo_search_mlp.log 2>&1
cd ..

cd attention
CUDA_VISIBLE_DEVICES=2,3 python bo_search_attn.py --use_effective_energy --normalize_objectives > bo_search_attn.log 2>&1
cd ..