# 3rdparty/ — Vendored dependencies

Kareus carries lightly modified copies of five upstream projects. All of them
are installed from source by [tests/data/install_deps.sh](../tests/data/install_deps.sh)
during the Docker build (see [Dockerfile](../Dockerfile)).

| Directory             | Upstream                                   | Vendored version | Tracked as            |
| --------------------- | ------------------------------------------ | ---------------- | --------------------- |
| `Megatron-LM/`        | https://github.com/NVIDIA/Megatron-LM      | megatron-core 0.12.1 | in-tree copy       |
| `NeMo/`               | https://github.com/NVIDIA/NeMo             | 2.3.1            | in-tree copy          |
| `TransformerEngine/`  | https://github.com/NVIDIA/TransformerEngine | 2.4.0           | in-tree copy (3 files patched) |
| `mscclpp/`            | https://github.com/microsoft/mscclpp       | 0.7.0            | in-tree copy          |
| `zeus/`               | https://github.com/ml-energy/zeus          | v0.15.1 (`8132e44`) | git submodule (see [`.gitmodules`](../.gitmodules)) |

Only `zeus/` is a real git submodule; the other four are checked into the
parent repository directly so reviewers do not need to clone anything beyond
this repo. The submodule is initialized automatically inside the container by
[tests/data/install_deps.sh](../tests/data/install_deps.sh), or manually with:

```bash
git submodule update --init 3rdparty/zeus
```

## Inspecting Kareus modifications

Each vendored tree contains small, surgical edits required by Kareus. To see
exactly what changed relative to the corresponding upstream release:

```bash
# Example: TransformerEngine 2.4.0
cd 3rdparty/TransformerEngine
git diff v2.4.0 -- .

# Example: Megatron-LM 0.12.1 (megatron-core)
cd 3rdparty/Megatron-LM
git diff core_v0.12.1 -- .
```

(You will need to fetch the upstream tags into a separate clone if they are
not already present; the in-tree copies do not carry upstream history.)

For `TransformerEngine` specifically, only three files are patched, and they
are listed explicitly in [tests/data/install_deps.sh](../tests/data/install_deps.sh)
(the install script copies them on top of the pip-installed package rather
than rebuilding TE from source):

- `transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py`
- `transformer_engine/pytorch/module/layernorm_linear.py`
- `transformer_engine/pytorch/ops/fuser.py`

For `zeus`, the submodule is pinned to the `paper consistent` branch which
sits a few commits past upstream `v0.15.1`; `git log v0.15.1..HEAD` inside
`3rdparty/zeus/` shows the Kareus-specific commits.
