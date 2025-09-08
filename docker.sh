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
    ruofanwu7/kareus-dev

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

# pip install -e ".[all]" 
# pip install -e . 

# zeus pip install -e . pip install '.[pfo-server]'