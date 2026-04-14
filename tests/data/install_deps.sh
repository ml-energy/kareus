#!/bin/bash
set -euo pipefail

KAREUS_DIR="/workspaces/Kareus"

# Allow git operations in this directory
git config --global --add safe.directory "$KAREUS_DIR"

# Initialize submodules that may not be checked out
git -C "$KAREUS_DIR" submodule update --init 3rdparty/zeus

# Remove any pip-installed megatron/nemo to avoid conflicts with source installs
rm -rf /usr/local/lib/python3.12/dist-packages/megatron* || true
rm -rf /usr/local/lib/python3.12/dist-packages/nemo* || true

# Install Megatron-LM from source
pip install "$KAREUS_DIR/3rdparty/Megatron-LM"

# Install nemo_toolkit[all] to pull in all NeMo dependencies
# (hydra-core, pytorch-lightning, omegaconf, transformers, etc.)
# This may install a newer nemo version which we override next.
pip install "nemo_toolkit[all]"

# Reinstall NeMo 2.3.1 from source to override the pip version
pip install "$KAREUS_DIR/3rdparty/NeMo"

# Install mscclpp from source
pip install "$KAREUS_DIR/3rdparty/mscclpp"

# Install zeus from source with pfo-server extras
pip install "$KAREUS_DIR/3rdparty/zeus"
pip install "$KAREUS_DIR/3rdparty/zeus[pfo-server]"

# Install remaining pip dependencies
pip install botorch mpi4py nemo_run ffmpeg

# Patch TransformerEngine with Kareus modifications
# (faster than reinstalling from source)
TE_DIR=$(python -c "import transformer_engine; import os; print(os.path.dirname(transformer_engine.__file__))")
cp "$KAREUS_DIR/3rdparty/TransformerEngine/transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py" \
   "$TE_DIR/pytorch/attention/dot_product_attention/dot_product_attention.py"
cp "$KAREUS_DIR/3rdparty/TransformerEngine/transformer_engine/pytorch/module/layernorm_linear.py" \
   "$TE_DIR/pytorch/module/layernorm_linear.py"
cp "$KAREUS_DIR/3rdparty/TransformerEngine/transformer_engine/pytorch/ops/fuser.py" \
   "$TE_DIR/pytorch/ops/fuser.py"
