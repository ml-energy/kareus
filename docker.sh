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

# mscclpp
cd ../mscclpp
python3 -m pip install .

# zeus 
cd ../zeus
pip install -e . 
pip install '.[pfo-server]'

cd ../../
rm /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py
cp 3rdparty/TransformerEngine/transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention/dot_product_attention/
rm /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/module/layernorm_linear.py
cp 3rdparty/TransformerEngine/transformer_engine/pytorch/module/layernorm_linear.py /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/module/
rm /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/ops/fuser.py
cp 3rdparty/TransformerEngine/transformer_engine/pytorch/ops/fuser.py /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/ops/

# /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/module/layernorm_linear.py
# allreduce
# /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py
# self.cp_comm_type = self.cp_comm_type if self.cp_comm_type is not None else cp_comm_type
# /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/ops/fuser.py
# ctx._saved_tensors_range = None

# # TransformerEngine
# cd ../TransformerEngine
# export NVTE_FRAMEWORK=pytorch 
# pip install --no-build-isolation -e .  

# # cargo
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

pip install botorch

pip install mpi4py

# huggingface-cli login 
# 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
# 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
# export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download --repo-type dataset ruofanwu/my-gpt_text_document --local-dir ./my-gpt_text_document
# cd ../
mkdir tests/simple_test/data
mv my-gpt_text_document/my-gpt_text_document.* tests/simple_test/data

# docker login -u ruofanwu7
# dckr_pat_DI1BqD7uAvlv4tNRX_H_fDxxYG4

# AWS
# https://github.com/ml-energy/vllm/commit/0ecb7b84140630f885c36bac0233023b8e9df7c0

sed -i 's|http://archive.ubuntu.com/ubuntu|http://us-east-1.ec2.archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list
apt-get update -o Acquire::Retries=3

curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.34.0.tar.gz \
&& tar -xf aws-efa-installer-1.34.0.tar.gz \
&& cd aws-efa-installer \
&& ./efa_installer.sh -y --skip-kmod --mpi=openmpi4 --no-verify

PATH="/opt/amazon/openmpi/bin:$PATH"
LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH"
NCCL_VERSION=2.20.5
cat <<'EOF' >> ~/.zshrc
# AWS OpenMPI/NCCL environment variables
export PATH="/opt/amazon/openmpi/bin:$PATH"
export LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH"
export NCCL_VERSION=2.20.5
EOF

cd $HOME \
&& git clone https://github.com/NVIDIA/nccl.git -b v${NCCL_VERSION}-1 \
&& cd nccl \
&& make -j64 src.build BUILDDIR=/usr/local

apt-get update && apt-get install -y autoconf libhwloc-dev wget
wget https://github.com/aws/aws-ofi-nccl/releases/download/v1.10.0-aws/aws-ofi-nccl-1.10.0-aws.tar.gz \
&& tar -xf aws-ofi-nccl-1.10.0-aws.tar.gz \
&& cd aws-ofi-nccl-1.10.0-aws \
&& ./configure --prefix=/usr/local --with-mpi=/opt/amazon/openmpi --with-libfabric=/opt/amazon/efa --with-cuda=/usr/local/cuda --with-nccl=/usr/local --enable-platform-aws \
&& make \
&& make install


# sudo chown -R ubuntu:ubuntu /home/ubuntu/.ssh
# sudo chown ubuntu:ubuntu /home/ubuntu

# sudo chmod 700 /home/ubuntu/.ssh
# sudo chmod 600 /home/ubuntu/.ssh/authorized_keys
# sudo chmod 755 /home/ubuntu

# sudo apt remove -y unattended-upgrades