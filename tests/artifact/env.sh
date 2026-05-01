#!/usr/bin/env bash
# Per-cluster environment for tests/artifact/ scripts.
#
# Edit the values below to match your 2-node A100 setup.  Every artifact
# run script and profiling helper sources this file, so changes here
# propagate to all of them.
#
# All variables are exported.  In-script defaults (e.g. MASTER_PORT,
# PFO_PORT) still apply for anything not set here.

# Master address: IP/hostname of node 0 (used by torchrun on both nodes
# and as the scp target from node 1).
export MASTER_ADDR="${MASTER_ADDR:-172.31.42.96}"

# Prefer the packaged AWS OFI NCCL plugin while avoiding /usr/local/lib,
# which can contain an older libnccl.so incompatible with this PyTorch build.
export LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/opt/amazon/aws-ofi-nccl/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-aws-ofi}"

# PyTorch 2.8 is built against NCCL >= 2.27 (needs ncclGroupSimulateEnd).
# aws.sh installs NCCL 2.20.5 into /usr/local/lib which shadows the container's
# 2.27.x via the ldconfig cache and breaks `import torch` with:
#   undefined symbol: ncclGroupSimulateEnd
# Force the correct libnccl.so.2 via LD_PRELOAD when present. Guarded so it's
# a no-op on hosts/containers that don't ship the newer NCCL there.
_KAREUS_NCCL_LIB="${KAREUS_NCCL_LIB:-/lib/x86_64-linux-gnu/libnccl.so.2}"
if [[ -f "${_KAREUS_NCCL_LIB}" ]] && [[ ":${LD_PRELOAD:-}:" != *":${_KAREUS_NCCL_LIB}:"* ]]; then
    export LD_PRELOAD="${_KAREUS_NCCL_LIB}${LD_PRELOAD:+:$LD_PRELOAD}"
fi
unset _KAREUS_NCCL_LIB

# Suppress noisy `pynvml` deprecation FutureWarning emitted by
# torch.cuda on import (PyTorch still imports the deprecated `pynvml`
# package internally). Scoped to that specific warning so other warnings
# are still surfaced.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:The pynvml package is deprecated:FutureWarning}"

# Node-1 -> node-0 result sync (scp/ssh from node 1).
export REMOTE_USER="${REMOTE_USER:-ubuntu}"
export REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/home/${REMOTE_USER:-ubuntu}/workspace/Kareus/tests/artifact}"
# Optional: path to the SSH private key node 1 uses to scp results to node 0.
export SSH_KEY_PATH="${SSH_KEY_PATH:-}"
