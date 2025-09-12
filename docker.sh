docker pull ruofanwu7/kareus-dev

docker run -it \
    --gpus=all \
    --ipc=host \
    --network=host \
    --name=ruofan-kareus \
    --privileged \
    -v /dev/shm:/dev/shm \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/.ssh:/root/.ssh \
    -v $HOME/workspace:/workspaces \
    ruofanwu7/kareus-dev2

# ssh-keygen -t ed25519 -C "ruofanw@umich.edu"
# sudo usermod -aG docker ubuntu

# git config --global --add safe.directory /workspaces/Kareus
# install cfuser
# pip install "numpy<2.0"

# pip uninstall megatron_energon
# pip uninstall megatron_core
# pip uninstall nemo_toolkit
# rm -r /usr/local/lib/python3.12/dist-packages/megatron*
# rm -r /usr/local/lib/python3.12/dist-packages/nemo*

# Megatron
# pip install -e .

# Nemo
# pip install -e . 

# /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/module/layernorm_linear.py

# mscclpp
# python3 -m pip install .

# zeus 
# pip install -e . 
# pip install '.[pfo-server]'

# huggingface-cli login 
# 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

# docker login -u ruofanwu7
# dckr_pat_DI1BqD7uAvlv4tNRX_H_fDxxYG4

# AWS
# https://github.com/ml-energy/vllm/commit/0ecb7b84140630f885c36bac0233023b8e9df7c0