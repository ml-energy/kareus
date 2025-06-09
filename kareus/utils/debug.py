from megatron.core import parallel_state
import pathlib
import torch

def save_tensors(tensors, name, id=1, save_dir="/workspaces/Kareus/tests/simple_test/compare_results"):
    save_dir = pathlib.Path(save_dir + f"/{id}")
    save_dir.mkdir(parents=True, exist_ok=True)
    rank = parallel_state.get_tensor_model_parallel_rank()
    filename = f"{name}_rank_{rank}.pt"
    save_path = save_dir / filename
    torch.save(tensors.detach().cpu(), save_path)
    print(f"Saved {name} tensor to {save_path}")