# 3rdparty/ — Vendored dependencies

Kareus carries lightly modified copies of five upstream projects.

| Directory             | Upstream                                   | Vendored version | Tracked as            |
| --------------------- | ------------------------------------------ | ---------------- | --------------------- |
| `Megatron-LM/`        | https://github.com/NVIDIA/Megatron-LM      | megatron-core 0.12.1 | in-tree copy       |
| `NeMo/`               | https://github.com/NVIDIA/NeMo             | 2.3.1            | in-tree copy          |
| `TransformerEngine/`  | https://github.com/NVIDIA/TransformerEngine | 2.4.0           | in-tree copy          |
| `mscclpp/`            | https://github.com/microsoft/mscclpp       | 0.7.0            | in-tree copy          |
| `zeus/`               | https://github.com/ml-energy/zeus          | v0.15.1 (`8132e44`) | git submodule      |


## Inspecting Kareus modifications

Each vendored tree contains small, surgical edits required by Kareus. Every
such edit is tagged in-source with one of two markers, so you can locate them
with a simple recursive search:

```bash
# From the repo root
grep -rn "Kareus Modifications" 3rdparty/   # file-level banner at the top of each modified file
grep -rn "\[Kareus\]"           3rdparty/   # inline tag on each modified line/block
```
