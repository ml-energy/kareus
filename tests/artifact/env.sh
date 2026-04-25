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
export MASTER_ADDR="${MASTER_ADDR:-10.0.0.1}"

# Node-1 -> node-0 result sync (scp/ssh from node 1).
export REMOTE_USER="${REMOTE_USER:-ubuntu}"
export REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-$HOME/workspace/Kareus/tests/artifact}"
export SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/ruofanw.pem}"
