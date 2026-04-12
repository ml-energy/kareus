import os
import random
import time
import argparse
import torch
import torch.distributed as dist
import sys
import csv
import traceback
import pynvml

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
# from overlap_test_attn import AttentionFuserTest
# from kareus.megatron.core.extensions.fusers.partition_fuser_profile import PartitionFuser
from zeus.monitor import ZeusMonitor

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
script_dir = Path("/workspaces/osdi/Kareus/")
style_file = script_dir / "paper.mplstyle"
if not style_file.exists():
    raise FileNotFoundError(f"Style file not found at: {style_file}")
plt.style.use(str(style_file))

AXIS_LABEL_FONT_SIZE = 44
TICK_FONT_SIZE = 36
LEGEND_FONT_SIZE = 28
mpl.rcParams["axes.labelsize"] = AXIS_LABEL_FONT_SIZE
mpl.rcParams["xtick.labelsize"] = TICK_FONT_SIZE
mpl.rcParams["ytick.labelsize"] = TICK_FONT_SIZE
mpl.rcParams["legend.fontsize"] = LEGEND_FONT_SIZE


def init_env(rank, world_size, master_port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"


def _spawn_entry(rank, world_size, args, master_port):
    init_env(rank, world_size, master_port)

    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    try:
        _run_study(rank, world_size, args)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        if rank == 0:
            pid = os.getpid()
            print(f"Killing process group {pid}")
            os.system(f'pkill -P {pid}')

        if dist.is_initialized():
            dist.destroy_process_group()
            print("Destroyed process group")


def _run_study(rank, world_size, args):
    tester = AttentionFuserTest(args, rank, world_size)
    # Build tensors and ops once
    test_tensors = tester.create_test_tensors()
    operations = tester.create_operations(test_tensors[-1])

    comp_ops = operations[:-1]
    allreduce_comm_op = operations[-1]
    
    attention_fuser = PartitionFuser(
        ops=comp_ops,
        comm_op_fwd=allreduce_comm_op,
        fuse_ops=False,
    )

    hidden_states, bias, residual, rotary_pos_emb, attention_mask, allreduce_inputs = test_tensors
    overlap_window = (args.overlap_start, args.overlap_end)
    sm_configs = (args.sm_num, args.block_size)
    current_stream = torch.cuda.current_stream()

    # Warmup passes to get iteration time
    torch.cuda.synchronize()
    dist.barrier()
    
    warmup_iterations = 20
    for _ in range(warmup_iterations):
        attention_fuser(
            hidden_states=hidden_states,
            bias=bias,
            residual=residual,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            comm_input=allreduce_inputs,
            comm_overlap_window=overlap_window,
            comm_sm_configs=sm_configs,
        )
    
    # Measure single iteration time
    # torch.cuda.synchronize()
    # dist.barrier()
    current_stream.synchronize()
    time_start = time.time()
    
    calibration_iters = 100
    for _ in range(calibration_iters):
        attention_fuser(
            hidden_states=hidden_states,
            bias=bias,
            residual=residual,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            comm_input=allreduce_inputs,
            comm_overlap_window=overlap_window,
            comm_sm_configs=sm_configs,
        )
    
    # torch.cuda.synchronize()
    # dist.barrier()
    current_stream.synchronize()
    time_end = time.time()
    
    iter_duration = (time_end - time_start) / calibration_iters
    if rank == 0:
        print(f"Single iteration duration: {iter_duration * 1000:.3f} ms")

    # Initialize ZeusMonitor and pynvml on rank 0
    monitor = None
    nvml_handles = []
    if rank == 0:
        gpu_indices = list(range(world_size))
        monitor = ZeusMonitor(gpu_indices=gpu_indices)
        # Initialize pynvml for temperature monitoring
        pynvml.nvmlInit()
        for i in range(world_size):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            nvml_handles.append(handle)

    # Cooldown study parameters
    target_duration = 5  # Fixed 5s duration
    # cooldown_times_sorted = [0.5] + list(range(1, 11)) + [15, 20, 25, 30]  # 1s-10s, then 15s, 20s, 25s, 30s
    cooldown_times_sorted = [0,]
    cooldown_times = list(reversed(cooldown_times_sorted))  # Measure from large to small cooldown
    repeats_per_cooldown = 10

    # Calculate iterations needed for target duration (rank 0 computes and broadcasts)
    if rank == 0:
        iterations = max(1, int(target_duration / iter_duration))
        dist_list = [iterations]
    else:
        dist_list = [None]
    dist.broadcast_object_list(dist_list, src=0)
    iterations = dist_list[0]
    
    if rank == 0:
        print(f"Target duration: {target_duration}s")
        print(f"Iterations per measurement: {iterations}")
        print(f"Expected actual duration: {iterations * iter_duration:.3f}s")

    # Results storage
    results = []
    
    # Setup CSV file and write header (only rank 0)
    output_dir = None
    csv_path = None
    if rank == 0:
        output_dir = f"logs/cooldown_study/tp{world_size}-bs{args.batch_size}-seq{args.seq_len}"
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "cooldown_energy_results.csv")
        
        # Write CSV header
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['cooldown_time', 'repeat', 'iterations', 'actual_time', 
                      'total_energy', 'energy_per_iter', 'time_per_iter', 'avg_temperature']
            for i in range(world_size):
                header.append(f'gpu{i}_energy')
                header.append(f'gpu{i}_energy_per_iter')
                header.append(f'gpu{i}_temperature')
            writer.writerow(header)
        print(f"CSV initialized: {csv_path}")

    for cooldown_time in cooldown_times:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Cooldown time: {cooldown_time}s")

        for repeat in range(repeats_per_cooldown):
            if rank == 0:
                print(f"  Repeat {repeat+1}/{repeats_per_cooldown}: Cooling down for {cooldown_time}s...")
            
            # Cooldown before each measurement
            torch.cuda.synchronize()
            dist.barrier()
            if cooldown_time > 0:
                time.sleep(cooldown_time)

            # Record temperature before measurement
            gpu_temperatures = []
            if rank == 0:
                for handle in nvml_handles:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_temperatures.append(temp)
            
            # Warmup after cooldown
            torch.cuda.synchronize()
            dist.barrier()
            for _ in range(20):
                attention_fuser(
                    hidden_states=hidden_states,
                    bias=bias,
                    residual=residual,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_mask=attention_mask,
                    comm_input=allreduce_inputs,
                    comm_overlap_window=overlap_window,
                    comm_sm_configs=sm_configs,
                )
            
            if rank == 0:
                monitor.begin_window("measurement")
            
            for _ in range(iterations):
                attention_fuser(
                    hidden_states=hidden_states,
                    bias=bias,
                    residual=residual,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_mask=attention_mask,
                    comm_input=allreduce_inputs,
                    comm_overlap_window=overlap_window,
                    comm_sm_configs=sm_configs,
                )
            
            # torch.cuda.synchronize()
            # dist.barrier()
            
            if rank == 0:
                result = monitor.end_window("measurement")
                actual_time = result.time
                total_energy = result.total_energy
                energy_per_iter = total_energy / iterations
                time_per_iter = actual_time / iterations
                
                # Per-GPU energy
                gpu_energies = [result.gpu_energy[i] for i in range(world_size)]
                gpu_energy_per_iter = [e / iterations for e in gpu_energies]
                
                avg_temperature = sum(gpu_temperatures) / len(gpu_temperatures) if gpu_temperatures else 0
                
                result_entry = {
                    'cooldown_time': cooldown_time,
                    'repeat': repeat,
                    'iterations': iterations,
                    'actual_time': actual_time,
                    'total_energy': total_energy,
                    'energy_per_iter': energy_per_iter,
                    'time_per_iter': time_per_iter,
                    'gpu_energies': gpu_energies,
                    'gpu_energy_per_iter': gpu_energy_per_iter,
                    'gpu_temperatures': gpu_temperatures,
                    'avg_temperature': avg_temperature,
                }
                results.append(result_entry)
                
                print(f"    -> time={actual_time:.3f}s, energy={total_energy:.3f}J, "
                      f"energy/iter={energy_per_iter*1000:.3f}mJ, avg_temp={avg_temperature:.1f}°C")
                
                # Append this result to CSV immediately
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [
                        result_entry['cooldown_time'],
                        result_entry['repeat'],
                        result_entry['iterations'],
                        result_entry['actual_time'],
                        result_entry['total_energy'],
                        result_entry['energy_per_iter'],
                        result_entry['time_per_iter'],
                        result_entry['avg_temperature'],
                    ]
                    for i in range(world_size):
                        row.append(result_entry['gpu_energies'][i])
                        row.append(result_entry['gpu_energy_per_iter'][i])
                        row.append(result_entry['gpu_temperatures'][i])
                    writer.writerow(row)

    # Final summary (only rank 0)
    if rank == 0:
        print(f"\nAll results saved to: {csv_path}")
        
        # Generate plot
        plot_path = os.path.join(output_dir, "cooldown_energy_plot.png")
        generate_plot(results, plot_path, world_size)
        print(f"Plot saved to: {plot_path}")

    # Shutdown pynvml
    if rank == 0:
        pynvml.nvmlShutdown()

    torch.cuda.synchronize()
    dist.barrier()


def generate_plot(results, output_path, world_size):
    """Generate a plot showing energy vs cooldown time with error ranges and temperature."""
    # Organize data by cooldown time - only include integer cooldowns from 1 to 10
    all_cooldowns = sorted(set(r['cooldown_time'] for r in results))
    cooldown_times = [c for c in all_cooldowns if c >= 0 and c <= 10 and c == int(c)]
    
    # Collect energy per iteration and temperature for each cooldown time
    energy_per_iter_by_cooldown = {c: [] for c in cooldown_times}
    temperature_by_cooldown = {c: [] for c in cooldown_times}
    
    for r in results:
        c = r['cooldown_time']
        if c not in cooldown_times:
            continue
        energy_per_iter_by_cooldown[c].append(r['energy_per_iter'])
        temperature_by_cooldown[c].append(r.get('avg_temperature', 0))
    
    # Calculate energy statistics (for summary)
    means_per_iter = [np.mean(energy_per_iter_by_cooldown[c]) for c in cooldown_times]
    stds_per_iter = [np.std(energy_per_iter_by_cooldown[c]) for c in cooldown_times]
    mins_per_iter = [np.min(energy_per_iter_by_cooldown[c]) for c in cooldown_times]
    maxs_per_iter = [np.max(energy_per_iter_by_cooldown[c]) for c in cooldown_times]
    
    # Calculate temperature statistics (only average)
    means_temp = [np.mean(temperature_by_cooldown[c]) for c in cooldown_times]
    
    # Create single figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7.5))
    ax2 = ax1.twinx()
    
    # Prepare data for violin plot
    violin_data = [energy_per_iter_by_cooldown[c] for c in cooldown_times]
    x_positions = list(range(1, len(cooldown_times) + 1))
    
    # Plot energy per iteration (left y-axis) with violin plot
    parts = ax1.violinplot(violin_data, positions=x_positions, widths=0.7,
                           showmeans=True, showmedians=False, showextrema=True)
    
    # Style the violin plot
    for pc in parts['bodies']:
        pc.set_facecolor('gray')
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)
        pc.set_linewidth(1.5)
    
    # Style the mean and extrema lines
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(1.5)
    parts['cmins'].set_color('black')
    parts['cmins'].set_linewidth(1.5)
    parts['cmaxes'].set_color('black')
    parts['cmaxes'].set_linewidth(1.5)
    parts['cbars'].set_color('black')
    parts['cbars'].set_linewidth(1.2)
    
    # Plot temperature (right y-axis) - average only
    ax2.plot(x_positions, means_temp, 's-', linewidth=3, markersize=10,
             color='#F18F01', label='Avg Temperature', zorder=4)
    
    # Configure left y-axis (Energy)
    ax1.set_xlabel('Cooldown Duration (s)')
    ax1.set_ylabel('Energy (J)')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([str(int(c)) for c in cooldown_times])
    ax1.set_xlim(0.5, len(cooldown_times) + 0.5)
    ax1.set_ylim(4.19, 4.81)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Configure right y-axis (Temperature) - use orange
    ax2.set_ylabel('Pre-Measurement\nGPU Temperature (°C)', color='#F18F01')
    # ax2.yaxis.set_label_coords(1.11, 0.5)  # Move label down
    ax2.tick_params(axis='y', labelcolor='#F18F01', colors='#F18F01')
    
    # Set spine linewidths and colors
    ax1.spines["left"].set_linewidth(1.2)
    ax1.spines["bottom"].set_linewidth(1.2)
    ax2.spines["right"].set_linewidth(1.2)
    ax2.spines["right"].set_color('#F18F01')
    ax2.spines["top"].set_linewidth(1.2)
    
    # No legend on main plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"{'Cooldown':>10} | {'Mean (J)':>12} | {'Std (J)':>10} | {'CV (%)':>8} | {'Range (J)':>15}")
    print("-"*80)
    for i, c in enumerate(cooldown_times):
        cv = (stds_per_iter[i] / means_per_iter[i]) * 100 if means_per_iter[i] > 0 else 0
        range_str = f"{mins_per_iter[i]:.4f}-{maxs_per_iter[i]:.4f}"
        print(f"{c:>10} | {means_per_iter[i]:>12.5f} | {stds_per_iter[i]:>10.5f} | {cv:>8.2f} | {range_str:>15}")
    print("="*80)


def load_results_from_csv(csv_path, world_size):
    """Load results from existing CSV file."""
    results = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = {
                'cooldown_time': float(row['cooldown_time']),
                'repeat': int(row['repeat']),
                'iterations': int(row['iterations']),
                'actual_time': float(row['actual_time']),
                'total_energy': float(row['total_energy']),
                'energy_per_iter': float(row['energy_per_iter']),
                'time_per_iter': float(row['time_per_iter']),
                'avg_temperature': float(row.get('avg_temperature', 0)),
                'gpu_energies': [float(row[f'gpu{i}_energy']) for i in range(world_size)],
                'gpu_energy_per_iter': [float(row[f'gpu{i}_energy_per_iter']) for i in range(world_size)],
                'gpu_temperatures': [float(row.get(f'gpu{i}_temperature', 0)) for i in range(world_size)],
            }
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="Study influence of cooldown time on energy readings")
    parser.add_argument("--world_size", "-w", type=int, default=8)
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--seq_len", "-s", type=int, default=4096)
    parser.add_argument("--overlap_start", type=int, default=0)
    parser.add_argument("--overlap_end", type=int, default=-1)
    parser.add_argument("--sm_num", "-n", type=int, default=12)
    parser.add_argument("--block_size", "-t", type=int, default=1024)
    args = parser.parse_args()

    # Check if CSV already exists
    output_dir = f"logs/cooldown_study/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}"
    csv_path = os.path.join(output_dir, "cooldown_energy_results.csv")
    
    if os.path.exists(csv_path):
        print("="*60)
        print("CSV file found, loading data and generating plots...")
        print(f"CSV path: {csv_path}")
        print("="*60)
        
        results = load_results_from_csv(csv_path, args.world_size)
        print(f"Loaded {len(results)} data points")
        
        # Generate plots
        plot_path = os.path.join(output_dir, "cooldown_energy_plot.png")
        generate_plot(results, plot_path, args.world_size)
        print(f"Plot saved to: {plot_path}")
        return

    # Calculate estimated time
    cooldown_times = list(range(1, 11)) + [15, 20, 25, 30]
    total_cooldown = sum(cooldown_times) * 10  # 10 repeats each
    estimated_minutes = (total_cooldown + 14 * 10 * 5) / 60  # cooldown + measurement time

    print("="*60)
    print("Cooldown-Energy Study for Attention Fuser")
    print("="*60)
    print(f"World size: {args.world_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Overlap window: ({args.overlap_start}, {args.overlap_end})")
    print(f"SM config: (sm_num={args.sm_num}, block_size={args.block_size})")
    print(f"Fixed duration: 5s")
    print(f"Cooldown times to test: 1s-10s, 15s, 20s, 25s, 30s")
    print(f"Repeats per cooldown: 10")
    print(f"Estimated total time: ~{estimated_minutes:.0f} minutes")
    print("="*60)

    from torch.multiprocessing import spawn
    spawn(
        _spawn_entry,
        args=(args.world_size, args, 9003),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()

