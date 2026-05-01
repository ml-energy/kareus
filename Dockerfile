# syntax=docker/dockerfile:1.7
# =============================================================================
# Kareus Artifact Docker Image
# =============================================================================
#
# Base: nvcr.io/nvidia/pytorch:25.06-py3 (Python 3.12, CUDA, PyTorch, TE,
#       Megatron-core, Apex, FlashAttention pre-installed).
#
# This Dockerfile bakes the dependencies installed by
# `tests/data/install_deps.sh` into the image WITHOUT copying the repo into
# image layers. The repo is exposed to the build via a BuildKit bind mount,
# pip installs from source land in /usr/local/lib/python3.12/dist-packages,
# the TE patch files are cp'd into the same site-packages tree, and the
# bind-mounted source is dropped at the end of the RUN.
#
# BUILD (from repo root):
#   DOCKER_BUILDKIT=1 docker build -t kareus-artifact:latest .
#
# RUN (mount the repo at the same path the script uses):
#   docker run -it \
#     --gpus=all --ipc=host --network=host --privileged \
#     --name=kareus-artifact \
#     -v /dev/shm:/dev/shm \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     -v $(pwd):/workspaces/Kareus \
#     kareus-artifact:latest
#
# NOTE: Before building, ensure the four vendored 3rdparty trees are present:
#   ls 3rdparty/   # must show Megatron-LM NeMo TransformerEngine mscclpp zeus
# Only `zeus` is a real submodule (initialized by install_deps.sh itself);
# the others must already be checked out on the host.
# =============================================================================

FROM nvcr.io/nvidia/pytorch:25.06-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV KAREUS_DIR=/workspaces/Kareus

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ffmpeg libsm6 libxext6 libglib2.0-0 \
        libspdlog-dev nlohmann-json3-dev pigz zsh \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,source=.,target=/workspaces/Kareus,rw \
    --mount=type=cache,target=/root/.cache/pip \
    bash /workspaces/Kareus/tests/data/install_deps.sh

# ---------------------------
# zsh + oh-my-zsh + plugins 
# ---------------------------
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git \
        ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k \
 && git clone --depth=1 https://github.com/zsh-users/zsh-autosuggestions \
        ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
 && git clone --depth=1 https://github.com/zsh-users/zsh-syntax-highlighting.git \
        ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

RUN sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc \
 && echo 'fpath+=${ZSH_CUSTOM:-${ZSH:-~/.oh-my-zsh}/custom}/plugins/zsh-completions/src' >> ~/.zshrc

ENV SHELL=/bin/zsh
WORKDIR /workspaces/Kareus
CMD ["zsh"]
