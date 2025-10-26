# docker pull ruofanwu7/kareus-dev
# docker pull ruofanwu7/kareus-dev

# # ssh-keygen -t ed25519 -C "ruofanw@umich.edu"
# # sudo usermod -aG docker ubuntu
# # workspace
# # git clone git@github.com:SymbioticLab/Kareus.git

# docker run -it \
#     --gpus=all \
#     --ipc=host \
#     --network=host \
#     --name=ruofan-kareus \
#     --privileged \
#     -v /dev/shm:/dev/shm \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     -v $HOME/.ssh:/root/.ssh \
#     -v $HOME/workspace:/workspaces \
#     ruofanwu7/kareus-dev

git config --global --add safe.directory /workspaces/Kareus
# # ssh-keygen -t ed25519 -C "ruofanw@umich.edu"
# # sudo usermod -aG docker ubuntu
# # workspace
# # git clone git@github.com:SymbioticLab/Kareus.git

# docker run -it \
#     --gpus=all \
#     --ipc=host \
#     --network=host \
#     --name=ruofan-kareus \
#     --privileged \
#     -v /dev/shm:/dev/shm \
#     -v /dev/infiniband:/dev/infiniband \
#     -v /sys/class/infiniband:/sys/class/infiniband \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     -v $HOME/.ssh:/root/.ssh \
#     -v $HOME/workspace:/workspaces \
#     ruofanwu7/kareus-dev

# git config --global --add safe.directory /workspaces/Kareus
# install cfuser
# pip install "numpy<2.0"

# pip uninstall megatron_energon
# pip uninstall megatron_core
# pip uninstall nemo_toolkit
rm -r /usr/local/lib/python3.12/dist-packages/megatron*
rm -r /usr/local/lib/python3.12/dist-packages/nemo*
# rm -r /usr/local/lib/python3.12/dist-packages/transformer_engine*

# Megatron
cd 3rdparty/Megatron-LM
pip install -e .

# Nemo
cd ../NeMo
pip install -e . 

# /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/module/layernorm_linear.py
# allreduce
# /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py
# self.cp_comm_type = self.cp_comm_type if self.cp_comm_type is not None else cp_comm_type
# /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/ops/fuser.py
# ctx._saved_tensors_range = None

# mscclpp
cd ../mscclpp
python3 -m pip install .

# zeus 
cd ../zeus
pip install -e . 
pip install '.[pfo-server]'

# # TransformerEngine
# cd ../TransformerEngine
# export NVTE_FRAMEWORK=pytorch 
# pip install --no-build-isolation -e .  

# cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

pip install botorch

pip install mpi4py

# huggingface-cli login 
# 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
# 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
# export HF_HUB_ENABLE_HF_TRANSFER=1
# huggingface-cli download --repo-type dataset ruofanwu/my-gpt_text_document \
#   --local-dir ./my-gpt_text_document

# docker login -u ruofanwu7
# dckr_pat_DI1BqD7uAvlv4tNRX_H_fDxxYG4

# AWS
# https://github.com/ml-energy/vllm/commit/0ecb7b84140630f885c36bac0233023b8e9df7c0