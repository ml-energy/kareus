import torch
import time
import requests
import subprocess
import os
import signal
import psutil


def assert_close_with_threshold(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    mismatch_threshold_pct: float = 0.1,  # 0.1% by default
    equal_nan: bool = False,
    msg: str = "",
) -> None:
    close = torch.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)
    mismatch_mask = ~close  # Boolean tensor showing mismatched elements

    # print(f"mismatch_mask shape: {mismatch_mask.int().shape}")
    num_mismatched = mismatch_mask.int().sum().item()
    total_elements = actual.numel()
    mismatch_percentage = (num_mismatched / total_elements) * 100

    if mismatch_percentage > mismatch_threshold_pct:
        # Print shape and distribution of mismatches
        error_msg = f"\nTensor shape: {actual.shape}\n"
        error_msg += f"Mismatched elements: {num_mismatched} / {total_elements} ({mismatch_percentage:.3f}%)\n"
        error_msg += f"Threshold allowed: {mismatch_threshold_pct}%\n"

        # # Show distribution of mismatches across dimensions
        # print(f"mismatch_mask: {mismatch_mask}")
        # for dim in range(len(actual.shape)):
        #     mismatches_per_slice = mismatch_mask.sum(
        #         dim=tuple(i for i in range(len(actual.shape)) if i != dim)
        #     )
        #     error_msg += (
        #         f"\nMismatches along dimension {dim} (shape={actual.shape[dim]}):\n"
        #     )
        #     error_msg += f"Min: {mismatches_per_slice.min().item()}, Max: {mismatches_per_slice.max().item()}\n"
        #     error_msg += f"Distribution: {mismatches_per_slice.tolist()}\n"

        # Find examples of largest differences
        abs_diff = torch.abs(actual - expected)
        rel_diff = abs_diff / (torch.abs(expected) + atol)

        max_abs_diff = abs_diff.max()
        max_rel_diff = rel_diff.max()

        # Get indices of maximum differences
        max_abs_indices = torch.where(abs_diff == max_abs_diff)
        max_rel_indices = torch.where(rel_diff == max_rel_diff)

        # Check if indices exist before trying to access them
        if len(max_abs_indices) > 0 and len(max_abs_indices[0]) > 0:
            error_msg += f"\nGreatest absolute difference: {max_abs_diff.item():.6f} at indices {tuple(idx[0].item() for idx in max_abs_indices)}\n"
        else:
            error_msg += f"\nGreatest absolute difference: {max_abs_diff.item():.6f}\n"

        if len(max_rel_indices) > 0 and len(max_rel_indices[0]) > 0:
            error_msg += f"Greatest relative difference: {max_rel_diff.item():.6f} at indices {tuple(idx[0].item() for idx in max_rel_indices)}\n"
        else:
            error_msg += f"Greatest relative difference: {max_rel_diff.item():.6f}\n"

        if msg:
            error_msg = f"{msg}\n{error_msg}"

        raise AssertionError(error_msg)


def assert_close(actual, expected, rtol=1e-05, atol=1e-08, equal_nan=False, msg=""):
    assert_close_with_threshold(actual, expected, rtol, atol, 0.1, equal_nan, msg)


def popen_launch_server(
    model: str,
    nnodes: int,
    nproc_per_node: int,
    host: str,
    port: int,
    output_type: str,
    ulysses_degree: int,
    ring_degree: int,
    timeout: float = 30,
):
    command = [
        "python3",
        "-m",
        "cfuser.server.launcher",
        "--nnode",
        str(nnodes),
        "--nproc_per_node",
        str(nproc_per_node),
        "--master_addr",
        host,
        "--master_port",
        str(port),
        "--",
        "--output_type",
        output_type,
        "--model",
        model,
        "--ulysses_degree",
        str(ulysses_degree),
        "--ring_degree",
        str(ring_degree),
    ]

    process = subprocess.Popen(command, stdout=None, stderr=None)

    base_url = f"http://{host}:{port}"

    start_time = time.time()
    with requests.Session() as session:
        while time.time() - start_time < timeout:
            try:
                response = session.get(
                    f"{base_url}/health",
                )
                if response.status_code == 200:
                    return process
            except requests.RequestException:
                pass
            time.sleep(10)
    raise TimeoutError("Server failed to start within the timeout period.")


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = 0):
    """Kill the process and all its child processes."""
    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass
