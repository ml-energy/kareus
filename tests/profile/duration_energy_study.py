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

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from overlap_test_attn import AttentionFuserTest
from kareus.megatron.core.extensions.fusers.partition_fuser_profile import PartitionFuser
from zeus.monitor import ZeusMonitor

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from pathlib import Path
script_dir = Path("/workspaces/osdi/Kareus/")
style_file = script_dir / "paper.mplstyle"
if not style_file.exists():
    raise FileNotFoundError(f"Style file not found at: {style_file}")
plt.style.use(str(style_file))

AXIS_LABEL_FONT_SIZE = 50
TICK_FONT_SIZE = 40
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

    # Duration study parameters
    target_durations = [0.5] + list(range(1, 11))  # 0.5s to 10s
    # target_durations = list(range(8, 11))  # 1s to 10s
    # target_durations = [0.5, 6, 8]
    repeats_per_duration = 10
    cooldown_time = 10  # 1 minute cooldown

    # Results storage
    results = []
    
    # Setup CSV file and write header (only rank 0)
    output_dir = None
    csv_path = None
    if rank == 0:
        output_dir = f"logs/duration_study/tp{world_size}-bs{args.batch_size}-seq{args.seq_len}"
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "duration_energy_results.csv")
        
        # Write CSV header
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['target_duration', 'repeat', 'iterations', 'actual_time', 
                      'total_energy', 'energy_per_iter', 'time_per_iter', 'avg_temperature']
            for i in range(world_size):
                header.append(f'gpu{i}_energy')
                header.append(f'gpu{i}_energy_per_iter')
                header.append(f'gpu{i}_temperature')
            writer.writerow(header)
        print(f"CSV initialized: {csv_path}")

    for target_duration in target_durations:
        # Calculate iterations needed for this duration (rank 0 computes and broadcasts)
        if rank == 0:
            iterations = max(1, int(target_duration / iter_duration))
            dist_list = [iterations]
        else:
            dist_list = [None]
        dist.broadcast_object_list(dist_list, src=0)
        iterations = dist_list[0]
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Target duration: {target_duration}s")
            print(f"Iterations to run: {iterations}")
            print(f"Expected actual duration: {iterations * iter_duration:.3f}s")

        for repeat in range(repeats_per_duration):
            if rank == 0:
                print(f"  Repeat {repeat+1}/{repeats_per_duration}: Cooling down for {cooldown_time}s...")
            
            # Cooldown before each measurement
            torch.cuda.synchronize()
            dist.barrier()
            time.sleep(cooldown_time)
            
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
            # torch.cuda.synchronize()
            # dist.barrier()
            
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
                
                # Record temperature after measurement
                gpu_temperatures = []
                for handle in nvml_handles:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_temperatures.append(temp)
                avg_temperature = sum(gpu_temperatures) / len(gpu_temperatures) if gpu_temperatures else 0
                
                result_entry = {
                    'target_duration': target_duration,
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
                        result_entry['target_duration'],
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
        
        # # Print progress after each target duration
        # if rank == 0:
        #     print(f"  Data for target_duration={target_duration}s saved to CSV")
        # dist.barrier()

    # Final summary (only rank 0)
    if rank == 0:
        print(f"\nAll results saved to: {csv_path}")
        
        # Generate plot
        plot_path = os.path.join(output_dir, "duration_energy_plot.png")
        generate_plot(results, plot_path, world_size)
        print(f"Plot saved to: {plot_path}")

    # Shutdown pynvml
    if rank == 0:
        pynvml.nvmlShutdown()

    torch.cuda.synchronize()
    dist.barrier()


def generate_plot(results, output_path, world_size):
    """Generate a plot showing energy vs duration with error ranges and temperature."""
    # Organize data by duration - only include integer durations from 1 to 10
    all_durations = sorted(set(r['target_duration'] for r in results))
    durations = [d for d in all_durations if d >= 1 and d == int(d)]
    
    # Collect energy per iteration and temperature for each duration
    energy_per_iter_by_duration = {d: [] for d in durations}
    temperature_by_duration = {d: [] for d in durations}
    
    for r in results:
        d = r['target_duration']
        if d not in durations:
            continue
        energy_per_iter_by_duration[d].append(r['energy_per_iter'])
        temperature_by_duration[d].append(r['avg_temperature'])
    
    # Calculate energy statistics
    means_per_iter = [np.mean(energy_per_iter_by_duration[d]) for d in durations]
    stds_per_iter = [np.std(energy_per_iter_by_duration[d]) for d in durations]
    mins_per_iter = [np.min(energy_per_iter_by_duration[d]) for d in durations]
    maxs_per_iter = [np.max(energy_per_iter_by_duration[d]) for d in durations]
    
    # Calculate temperature statistics (only average)
    means_temp = [np.mean(temperature_by_duration[d]) for d in durations]
    
    # Get number of repeats for legend
    num_repeats = len([r for r in results if r['target_duration'] == durations[0]])
    
    # Create single figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7.5))
    ax2 = ax1.twinx()
    
    box_width = 0.6
    
    # Plot energy per iteration (left y-axis) with custom box style
    for i, d in enumerate(durations):
        x = i + 1  # 1-indexed position
        mean_val = means_per_iter[i]
        std_val = stds_per_iter[i]
        min_val = mins_per_iter[i]
        max_val = maxs_per_iter[i]
        
        # Draw min-max vertical line
        ax1.plot([x, x], [min_val, max_val], color='black', linewidth=3, zorder=1)
        # Draw min/max caps
        ax1.plot([x - box_width/4, x + box_width/4], [min_val, min_val], color='black', linewidth=3, zorder=1)
        ax1.plot([x - box_width/4, x + box_width/4], [max_val, max_val], color='black', linewidth=3, zorder=1)
        # Draw mean±std box
        rect = Rectangle((x - box_width/2, mean_val - std_val), box_width, 2 * std_val,
                         facecolor='gray', edgecolor='black', alpha=0.6, linewidth=2.5, zorder=2)
        ax1.add_patch(rect)
        # Draw mean line
        ax1.plot([x - box_width/2, x + box_width/2], [mean_val, mean_val], color='black', linewidth=3, zorder=3)
    
    # Plot temperature (right y-axis) - average only
    x_positions = list(range(1, len(durations) + 1))
    ax2.plot(x_positions, means_temp, 's-', linewidth=3, markersize=10,
             color='#F18F01', label='Avg Temperature', zorder=4)
    
    # Configure left y-axis (Energy)
    ax1.set_xlabel('Measurement Window (s)')
    ax1.set_ylabel('Energy (J)')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([str(int(d)) for d in durations])
    ax1.set_xlim(0.5, len(durations) + 0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Configure right y-axis (Temperature)
    ax2.set_ylabel('Post-Measurement\nGPU Temperature (°C)')
    ax2.yaxis.set_label_coords(1.11, 0.42)  # Move label down
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Set spine linewidths
    ax1.spines["left"].set_linewidth(1.2)
    ax1.spines["bottom"].set_linewidth(1.2)
    ax2.spines["right"].set_linewidth(1.2)
    ax2.spines["top"].set_linewidth(1.2)
    
    # No legend on main plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close()
    
    # Create separate legend figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    # Custom handler for min-max with caps
    class MinMaxHandler:
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height
            # Vertical line
            vline = Line2D([x0 + width/2, x0 + width/2], [y0, y0 + height],
                          color='black', linewidth=3)
            # Top cap
            top_cap = Line2D([x0 + width/4, x0 + 3*width/4], [y0 + height, y0 + height],
                            color='black', linewidth=3)
            # Bottom cap
            bottom_cap = Line2D([x0 + width/4, x0 + 3*width/4], [y0, y0],
                               color='black', linewidth=3)
            handlebox.add_artist(vline)
            handlebox.add_artist(top_cap)
            handlebox.add_artist(bottom_cap)
            return vline
    
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', alpha=0.6, label='Energy mean ± std'),
        Line2D([0], [0], color='black', linewidth=3, label='mean'),
        Line2D([0], [0], color='black', linewidth=3, marker='_', markersize=12, label='min-max'),
        Line2D([0], [0], color='#F18F01', marker='s', markersize=10, linewidth=3, label='Temperature'),
    ]
    
    # Create legend-only figure
    fig_legend = plt.figure(figsize=(10, 0.5))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')
    
    legend = ax_legend.legend(
        handles=legend_elements,
        loc='center',
        ncol=4,
        frameon=False,
        handler_map={legend_elements[2]: MinMaxHandler()}
    )
    
    legend_path = output_path.replace('.png', '_legend.png')
    fig_legend.savefig(legend_path, dpi=150, bbox_inches='tight')
    fig_legend.savefig(legend_path.replace('.png', '.pdf'), bbox_inches='tight')
    fig_legend.savefig(legend_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig_legend)
    print(f"Legend saved to: {legend_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"{'Duration':>10} | {'Mean (mJ)':>12} | {'Std (mJ)':>10} | {'CV (%)':>8} | {'Range (mJ)':>15}")
    print("-"*80)
    for i, d in enumerate(durations):
        cv = (stds_per_iter[i] / means_per_iter[i]) * 100 if means_per_iter[i] > 0 else 0
        range_str = f"{mins_per_iter[i]:.2f}-{maxs_per_iter[i]:.2f}"
        print(f"{d:>10} | {means_per_iter[i]:>12.3f} | {stds_per_iter[i]:>10.3f} | {cv:>8.2f} | {range_str:>15}")
    print("="*80)


def load_results_from_csv(csv_path, world_size):
    """Load results from existing CSV file."""
    results = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = {
                'target_duration': float(row['target_duration']),
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
    parser = argparse.ArgumentParser(description="Study influence of measurement duration on energy readings")
    parser.add_argument("--world_size", "-w", type=int, default=8)
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--seq_len", "-s", type=int, default=4096)
    parser.add_argument("--overlap_start", type=int, default=0)
    parser.add_argument("--overlap_end", type=int, default=-1)
    parser.add_argument("--sm_num", "-n", type=int, default=12)
    parser.add_argument("--block_size", "-t", type=int, default=1024)
    args = parser.parse_args()

    # Check if CSV already exists
    output_dir = f"logs/duration_study/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}"
    csv_path = os.path.join(output_dir, "duration_energy_results.csv")
    
    if os.path.exists(csv_path):
        print("="*60)
        print("CSV file found, loading data and generating plots...")
        print(f"CSV path: {csv_path}")
        print("="*60)
        
        results = load_results_from_csv(csv_path, args.world_size)
        print(f"Loaded {len(results)} data points")
        
        # Generate plots
        plot_path = os.path.join(output_dir, "duration_energy_plot.png")
        generate_plot(results, plot_path, args.world_size)
        print(f"Plot saved to: {plot_path}")
        return

    print("="*60)
    print("Duration-Energy Study for Attention Fuser")
    print("="*60)
    print(f"World size: {args.world_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Overlap window: ({args.overlap_start}, {args.overlap_end})")
    print(f"SM config: (sm_num={args.sm_num}, block_size={args.block_size})")
    print(f"Durations to test: 0.5s to 10s")
    print(f"Repeats per duration: 30")
    print(f"Cooldown before each repeat: 30s")
    print(f"Estimated total time: ~57 minutes")
    print("="*60)

    from torch.multiprocessing import spawn
    spawn(
        _spawn_entry,
        args=(args.world_size, args, 9002),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()

