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
    torch.cuda.synchronize()
    dist.barrier()
    time_start = time.time()
    
    calibration_iters = 50
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
    
    torch.cuda.synchronize()
    dist.barrier()
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
    cooldown_times_sorted = list(range(1, 11)) + [15, 20, 25, 30]  # 1s-10s, then 15s, 20s, 25s, 30s
    cooldown_times = list(reversed(cooldown_times_sorted))  # Measure from large to small cooldown
    repeats_per_cooldown = 10

    # Calculate iterations needed for target duration
    iterations = max(1, int(target_duration / iter_duration))
    if rank == 0:
        print(f"Target duration: {target_duration}s")
        print(f"Iterations per measurement: {iterations}")
        print(f"Expected actual duration: {iterations * iter_duration:.3f}s")

    # Results storage
    results = []

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
                
                results.append({
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
                })
                
                print(f"    -> time={actual_time:.3f}s, energy={total_energy:.3f}J, "
                      f"energy/iter={energy_per_iter*1000:.3f}mJ, avg_temp={avg_temperature:.1f}°C")

    # Save results to CSV (only rank 0)
    if rank == 0:
        # Sort results by cooldown_time (small to large) for CSV and plots
        results = sorted(results, key=lambda r: (r['cooldown_time'], r['repeat']))
        
        output_dir = f"logs/cooldown_study/tp{world_size}-bs{args.batch_size}-seq{args.seq_len}"
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, "cooldown_energy_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            header = ['cooldown_time', 'repeat', 'iterations', 'actual_time', 
                      'total_energy', 'energy_per_iter', 'time_per_iter', 'avg_temperature']
            for i in range(world_size):
                header.append(f'gpu{i}_energy')
                header.append(f'gpu{i}_energy_per_iter')
                header.append(f'gpu{i}_temperature')
            writer.writerow(header)
            
            # Data
            for r in results:
                row = [
                    r['cooldown_time'],
                    r['repeat'],
                    r['iterations'],
                    r['actual_time'],
                    r['total_energy'],
                    r['energy_per_iter'],
                    r['time_per_iter'],
                    r['avg_temperature'],
                ]
                for i in range(world_size):
                    row.append(r['gpu_energies'][i])
                    row.append(r['gpu_energy_per_iter'][i])
                    row.append(r['gpu_temperatures'][i])
                writer.writerow(row)
        
        print(f"\nResults saved to: {csv_path}")
        
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
    """Generate a plot showing energy vs cooldown time with error ranges."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Organize data by cooldown time
    cooldown_times = sorted(set(r['cooldown_time'] for r in results))
    
    # Collect energy per iteration and temperature for each cooldown time
    energy_per_iter_by_cooldown = {c: [] for c in cooldown_times}
    total_energy_by_cooldown = {c: [] for c in cooldown_times}
    temperature_by_cooldown = {c: [] for c in cooldown_times}
    
    for r in results:
        c = r['cooldown_time']
        energy_per_iter_by_cooldown[c].append(r['energy_per_iter'] * 1000)  # Convert to mJ
        total_energy_by_cooldown[c].append(r['total_energy'])
        temperature_by_cooldown[c].append(r.get('avg_temperature', 0))
    
    # Calculate statistics
    means_per_iter = [np.mean(energy_per_iter_by_cooldown[c]) for c in cooldown_times]
    stds_per_iter = [np.std(energy_per_iter_by_cooldown[c]) for c in cooldown_times]
    mins_per_iter = [np.min(energy_per_iter_by_cooldown[c]) for c in cooldown_times]
    maxs_per_iter = [np.max(energy_per_iter_by_cooldown[c]) for c in cooldown_times]
    
    means_total = [np.mean(total_energy_by_cooldown[c]) for c in cooldown_times]
    stds_total = [np.std(total_energy_by_cooldown[c]) for c in cooldown_times]
    mins_total = [np.min(total_energy_by_cooldown[c]) for c in cooldown_times]
    maxs_total = [np.max(total_energy_by_cooldown[c]) for c in cooldown_times]
    
    # Temperature statistics
    means_temp = [np.mean(temperature_by_cooldown[c]) for c in cooldown_times]
    stds_temp = [np.std(temperature_by_cooldown[c]) for c in cooldown_times]
    
    # Get number of repeats for legend
    num_repeats = len([r for r in results if r['cooldown_time'] == cooldown_times[0]])
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Energy per iteration vs Cooldown Time (with temperature on secondary y-axis)
    ax1 = axes[0]
    ax1.errorbar(cooldown_times, means_per_iter, yerr=stds_per_iter, 
                 fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
                 color='#2E86AB', ecolor='#A23B72', label=f'Energy: Mean ± Std (n={num_repeats})')
    ax1.fill_between(cooldown_times, mins_per_iter, maxs_per_iter, 
                     alpha=0.2, color='#2E86AB', label='Energy: Min-Max Range')
    ax1.set_xlabel('Cooldown Time (seconds)', fontsize=12)
    ax1.set_ylabel('Energy per Iteration (mJ)', fontsize=12, color='#2E86AB')
    ax1.tick_params(axis='y', labelcolor='#2E86AB')
    ax1.set_title('Energy per Iteration & Temperature vs Cooldown Time', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(cooldown_times)
    
    # Secondary y-axis for temperature
    ax1_temp = ax1.twinx()
    ax1_temp.plot(cooldown_times, means_temp, 's--', color='#E74C3C', linewidth=2, markersize=6, label='Avg Temperature')
    ax1_temp.fill_between(cooldown_times, 
                          [m - s for m, s in zip(means_temp, stds_temp)],
                          [m + s for m, s in zip(means_temp, stds_temp)],
                          alpha=0.15, color='#E74C3C')
    ax1_temp.set_ylabel('Temperature (°C)', fontsize=12, color='#E74C3C')
    ax1_temp.tick_params(axis='y', labelcolor='#E74C3C')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_temp.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    
    # Add annotation explaining the plot
    ax1.text(0.02, 0.98, 'Blue: energy per iteration\nRed: avg GPU temperature\nShaded: ±1 std',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Total Energy vs Cooldown Time
    ax2 = axes[1]
    ax2.errorbar(cooldown_times, means_total, yerr=stds_total,
                 fmt='s-', capsize=5, capthick=2, linewidth=2, markersize=8,
                 color='#F18F01', ecolor='#C73E1D', label=f'Mean ± Std (n={num_repeats})')
    ax2.fill_between(cooldown_times, mins_total, maxs_total,
                     alpha=0.2, color='#F18F01', label='Min-Max Range')
    ax2.set_xlabel('Cooldown Time (seconds)', fontsize=12)
    ax2.set_ylabel('Total Energy (J)', fontsize=12)
    ax2.set_title('Total Energy vs Cooldown Time', fontsize=14)
    ax2.legend(loc='best', title='Legend', fontsize=10, title_fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(cooldown_times)
    # Add annotation explaining the plot
    ax2.text(0.02, 0.98, 'Points: mean total energy\nError bars: ±1 standard deviation\nShaded: min-max range across repeats',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Impact of Cooldown Time on Energy Reading Stability', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create additional plot with custom box style (mean±std box, min-max line)
    box_plot_path = output_path.replace('.png', '_boxplot.png')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Custom box plot 1: Energy per iteration
    ax1 = axes[0]
    box_width = 0.6
    for i, c in enumerate(cooldown_times):
        x = i + 1  # 1-indexed position
        mean_val = means_per_iter[i]
        std_val = stds_per_iter[i]
        min_val = mins_per_iter[i]
        max_val = maxs_per_iter[i]
        
        # Draw min-max vertical line
        ax1.plot([x, x], [min_val, max_val], color='#2E86AB', linewidth=2, zorder=1)
        # Draw min/max caps
        ax1.plot([x - box_width/4, x + box_width/4], [min_val, min_val], color='#2E86AB', linewidth=2, zorder=1)
        ax1.plot([x - box_width/4, x + box_width/4], [max_val, max_val], color='#2E86AB', linewidth=2, zorder=1)
        # Draw mean±std box
        from matplotlib.patches import Rectangle
        rect = Rectangle((x - box_width/2, mean_val - std_val), box_width, 2 * std_val,
                         facecolor='#2E86AB', edgecolor='#1a5276', alpha=0.6, linewidth=1.5, zorder=2)
        ax1.add_patch(rect)
        # Draw mean line
        ax1.plot([x - box_width/2, x + box_width/2], [mean_val, mean_val], color='#E74C3C', linewidth=2, zorder=3)
    
    ax1.set_xlabel('Cooldown Time (seconds)', fontsize=12)
    ax1.set_ylabel('Energy per Iteration (mJ)', fontsize=12)
    ax1.set_title('Distribution of Energy per Iteration', fontsize=14)
    ax1.set_xticks(range(1, len(cooldown_times) + 1))
    ax1.set_xticklabels([str(c) for c in cooldown_times])
    ax1.set_xlim(0.5, len(cooldown_times) + 0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    # Add annotation
    ax1.text(0.02, 0.98, f'Box: mean ± std (n={num_repeats})\nRed line: mean\nVertical line: min-max range',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Custom box plot 2: Total Energy
    ax2 = axes[1]
    for i, c in enumerate(cooldown_times):
        x = i + 1
        mean_val = means_total[i]
        std_val = stds_total[i]
        min_val = mins_total[i]
        max_val = maxs_total[i]
        
        # Draw min-max vertical line
        ax2.plot([x, x], [min_val, max_val], color='#F18F01', linewidth=2, zorder=1)
        # Draw min/max caps
        ax2.plot([x - box_width/4, x + box_width/4], [min_val, min_val], color='#F18F01', linewidth=2, zorder=1)
        ax2.plot([x - box_width/4, x + box_width/4], [max_val, max_val], color='#F18F01', linewidth=2, zorder=1)
        # Draw mean±std box
        rect = Rectangle((x - box_width/2, mean_val - std_val), box_width, 2 * std_val,
                         facecolor='#F18F01', edgecolor='#c76d00', alpha=0.6, linewidth=1.5, zorder=2)
        ax2.add_patch(rect)
        # Draw mean line
        ax2.plot([x - box_width/2, x + box_width/2], [mean_val, mean_val], color='#E74C3C', linewidth=2, zorder=3)
    
    ax2.set_xlabel('Cooldown Time (seconds)', fontsize=12)
    ax2.set_ylabel('Total Energy (J)', fontsize=12)
    ax2.set_title('Distribution of Total Energy', fontsize=14)
    ax2.set_xticks(range(1, len(cooldown_times) + 1))
    ax2.set_xticklabels([str(c) for c in cooldown_times])
    ax2.set_xlim(0.5, len(cooldown_times) + 0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    # Add annotation
    ax2.text(0.02, 0.98, f'Box: mean ± std (n={num_repeats})\nRed line: mean\nVertical line: min-max range',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Energy Distribution by Cooldown Time', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(box_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Box plot saved to: {box_plot_path}")
    
    # Print summary statistics
    print("\n" + "="*95)
    print("SUMMARY STATISTICS")
    print("="*95)
    print(f"{'Cooldown':>10} | {'Mean (mJ)':>12} | {'Std (mJ)':>10} | {'CV (%)':>8} | {'Range (mJ)':>15} | {'Temp (°C)':>10}")
    print("-"*95)
    for i, c in enumerate(cooldown_times):
        cv = (stds_per_iter[i] / means_per_iter[i]) * 100 if means_per_iter[i] > 0 else 0
        range_str = f"{mins_per_iter[i]:.2f}-{maxs_per_iter[i]:.2f}"
        temp_str = f"{means_temp[i]:.1f}±{stds_temp[i]:.1f}"
        print(f"{c:>10} | {means_per_iter[i]:>12.3f} | {stds_per_iter[i]:>10.3f} | {cv:>8.2f} | {range_str:>15} | {temp_str:>10}")
    print("="*95)


def load_results_from_csv(csv_path, world_size):
    """Load results from existing CSV file."""
    results = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = {
                'cooldown_time': int(row['cooldown_time']),
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
    parser.add_argument("--world_size", "-w", type=int, default=2)
    parser.add_argument("--batch_size", "-b", type=int, default=4)
    parser.add_argument("--seq_len", "-s", type=int, default=4096)
    parser.add_argument("--overlap_start", type=int, default=0)
    parser.add_argument("--overlap_end", type=int, default=-1)
    parser.add_argument("--sm_num", "-n", type=int, default=6)
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

