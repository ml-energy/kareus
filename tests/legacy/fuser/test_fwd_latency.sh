cd attention
python test_attn_baseline.py

cd ../mlp
python test_mlp_baseline.py

cd ../prepost
python profile_loss.py
python profile_postprocess.py
