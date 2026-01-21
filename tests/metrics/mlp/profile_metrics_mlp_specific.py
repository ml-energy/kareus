import os
import time
import pynvml
from typing import List


# List of specific configurations to profile (forward)
# Each entry: (frequency_mhz, overlap_start, overlap_end, sm_num, block_size)
CONFIGS_TO_PROFILE_FORWARD = [
    (1350, 0, 6, 6, 1024),
    (900, 0, 6, 30, 1024),
    (1290, 0, 6, 6, 1024),
    (1290, 0, 6, 12, 1024),
    (960, 0, 6, 6, 1024),
]

# List of specific configurations to profile (backward)
# Each entry: (frequency_mhz, overlap_start, overlap_end, sm_num, block_size)
CONFIGS_TO_PROFILE_BACKWARD = [
    (1350, 0, 6, 6, 1024),
    (1290, 2, 6, 6, 1024),
    (960, 0, 6, 6, 1024),
]


def _set_gpu_frequency(target_freq_mhz: int, device_indices: List[int] | None = None) -> None:
    """Attempt to set application clocks via NVML (best effort).

    If device_indices is provided, only set those NVML indices; otherwise set all
    NVML-visible devices.
    """
    pynvml.nvmlInit()
    all_indices = list(range(pynvml.nvmlDeviceGetCount()))
    indices = device_indices if device_indices is not None else all_indices
    for i in indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, target_freq_mhz, target_freq_mhz)
    time.sleep(2)
    pynvml.nvmlShutdown()


def run_cmd(cmd_str):
    os.system(cmd_str)


def profile_configs(args, configs, target_indices, nvml_device_indices, mode="forward"):
    """
    Profile the given configurations.
    mode: "forward" or "backward"
    """
    script_name = "overlap_test_mlp_individual.py" if mode == "forward" else "overlap_test_mlp_individual_backward.py"
    
    for freq_mhz, overlap_start, overlap_end, sm_num, block_size in configs:
        frequency_str = str(freq_mhz)
        
        os.makedirs(f"profile_result/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}/{frequency_str}/{mode}", exist_ok=True)

        output_name = f"profile_{overlap_start}_{overlap_end}_{sm_num}_{block_size}"
        nsys_report = f"profile_result/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}/{frequency_str}/{mode}/{output_name}.nsys-rep"

        try:
            # Set GPU frequency before launching the profiling process
            print(f"[*] Setting GPU frequency to {freq_mhz} MHz...")
            _set_gpu_frequency(freq_mhz, nvml_device_indices)
            print(f"[✓] GPU frequency set to {freq_mhz} MHz")

            profile_cmd = [
                "nsys profile",
                "--gpu-metrics-devices", ",".join(target_indices),
                "--capture-range", "cudaProfilerApi",
                "--force-overwrite", "true",
                "-o", f"profile_result/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}/{frequency_str}/{mode}/{output_name}",
                "python", script_name,
                "--world_size", str(args.world_size),
                "--batch_size", str(args.batch_size),
                "--seq_len", str(args.seq_len),
                "--frequency", frequency_str,
                "--overlap_start", str(overlap_start),
                "--overlap_end", str(overlap_end),
                "--sm_num", str(sm_num),
                "--block_size", str(block_size),
            ]
            if not os.path.exists(nsys_report):
                run_cmd(" ".join(profile_cmd))
            print(f"[✓] nsys profiling done ({mode}). Output: {nsys_report}")
            print(f"    Config: freq={freq_mhz}, overlap=({overlap_start}, {overlap_end}), SM=({sm_num}, {block_size})")

        except Exception as e:
            print(f"[❌] Error processing {output_name} ({mode}): {e}")
            continue


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", "-w", type=int, default=8)
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--seq_len", "-s", type=int, default=4096)
    parser.add_argument("--mode", "-m", type=str, choices=["forward", "backward", "both"], default="both",
                        help="Profile mode: forward, backward, or both")
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible is not None and len(visible.strip()) > 0:
        vis_list = [x for x in visible.split(",") if x.strip() != ""]
        target_indices = vis_list
        # Convert to int for NVML device indices
        nvml_device_indices = [int(x) for x in vis_list]
    else:
        raise ValueError("CUDA_VISIBLE_DEVICES is not set")

    if args.mode in ["forward", "both"]:
        print("\n" + "="*60)
        print("Profiling FORWARD configurations")
        print("="*60)
        profile_configs(args, CONFIGS_TO_PROFILE_FORWARD, target_indices, nvml_device_indices, mode="forward")

    if args.mode in ["backward", "both"]:
        print("\n" + "="*60)
        print("Profiling BACKWARD configurations")
        print("="*60)
        profile_configs(args, CONFIGS_TO_PROFILE_BACKWARD, target_indices, nvml_device_indices, mode="backward")

