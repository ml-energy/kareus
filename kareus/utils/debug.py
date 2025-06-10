from megatron.core import parallel_state
import pathlib
import torch

def save_tensors(tensors, name, source="kareus", save_dir="/workspaces/Kareus/tests/simple_test/compare_results"):
    save_dir = pathlib.Path(save_dir + f"/{source}")
    save_dir.mkdir(parents=True, exist_ok=True)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    filename = f"{name}_tp_rank_{tp_rank}_pp_rank_{pp_rank}.pt"
    save_path = save_dir / filename
    torch.save(tensors.detach().cpu(), save_path)
    print(f"Saved {name} tensor to {save_path}")