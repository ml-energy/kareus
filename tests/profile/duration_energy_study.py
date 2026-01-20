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
    """Generate a plot showing energy vs duration with error ranges."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Organize data by duration
    durations = sorted(set(r['target_duration'] for r in results))
    
    # Collect energy per iteration for each duration
    energy_per_iter_by_duration = {d: [] for d in durations}
    total_energy_by_duration = {d: [] for d in durations}
    
    for r in results:
        d = r['target_duration']
        energy_per_iter_by_duration[d].append(r['energy_per_iter'] * 1000)  # Convert to mJ
        total_energy_by_duration[d].append(r['total_energy'])
    
    # Calculate statistics
    means_per_iter = [np.mean(energy_per_iter_by_duration[d]) for d in durations]
    stds_per_iter = [np.std(energy_per_iter_by_duration[d]) for d in durations]
    mins_per_iter = [np.min(energy_per_iter_by_duration[d]) for d in durations]
    maxs_per_iter = [np.max(energy_per_iter_by_duration[d]) for d in durations]
    
    means_total = [np.mean(total_energy_by_duration[d]) for d in durations]
    stds_total = [np.std(total_energy_by_duration[d]) for d in durations]
    mins_total = [np.min(total_energy_by_duration[d]) for d in durations]
    maxs_total = [np.max(total_energy_by_duration[d]) for d in durations]
    
    # Get number of repeats for legend
    num_repeats = len([r for r in results if r['target_duration'] == durations[0]])
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Energy per iteration vs Duration
    ax1 = axes[0]
    ax1.errorbar(durations, means_per_iter, yerr=stds_per_iter, 
                 fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
                 color='#2E86AB', ecolor='#A23B72', label=f'Mean ± Std (n={num_repeats})')
    ax1.fill_between(durations, mins_per_iter, maxs_per_iter, 
                     alpha=0.2, color='#2E86AB', label='Min-Max Range')
    ax1.set_xlabel('Measurement Duration (seconds)', fontsize=12)
    ax1.set_ylabel('Energy per Iteration (mJ)', fontsize=12)
    ax1.set_title('Energy per Iteration vs Measurement Duration', fontsize=14)
    ax1.legend(loc='best', title='Legend', fontsize=10, title_fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(durations)
    # Add annotation explaining the plot
    ax1.text(0.02, 0.98, 'Points: mean energy per iteration\nError bars: ±1 standard deviation\nShaded: min-max range across repeats',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Total Energy vs Duration
    ax2 = axes[1]
    ax2.errorbar(durations, means_total, yerr=stds_total,
                 fmt='s-', capsize=5, capthick=2, linewidth=2, markersize=8,
                 color='#F18F01', ecolor='#C73E1D', label=f'Mean ± Std (n={num_repeats})')
    ax2.fill_between(durations, mins_total, maxs_total,
                     alpha=0.2, color='#F18F01', label='Min-Max Range')
    ax2.set_xlabel('Measurement Duration (seconds)', fontsize=12)
    ax2.set_ylabel('Total Energy (J)', fontsize=12)
    ax2.set_title('Total Energy vs Measurement Duration', fontsize=14)
    ax2.legend(loc='best', title='Legend', fontsize=10, title_fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(durations)
    # Add annotation explaining the plot
    ax2.text(0.02, 0.98, 'Points: mean total energy\nError bars: ±1 standard deviation\nShaded: min-max range across repeats',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Impact of Measurement Duration on Energy Reading Stability', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create additional plot with custom box style (mean±std box, min-max line)
    box_plot_path = output_path.replace('.png', '_boxplot.png')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Custom box plot 1: Energy per iteration
    ax1 = axes[0]
    box_width = 0.6
    for i, d in enumerate(durations):
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
    
    ax1.set_xlabel('Measurement Duration (seconds)', fontsize=12)
    ax1.set_ylabel('Energy per Iteration (mJ)', fontsize=12)
    ax1.set_title('Distribution of Energy per Iteration', fontsize=14)
    ax1.set_xticks(range(1, len(durations) + 1))
    ax1.set_xticklabels([str(d) for d in durations])
    ax1.set_xlim(0.5, len(durations) + 0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    # Add annotation
    ax1.text(0.02, 0.98, f'Box: mean ± std (n={num_repeats})\nRed line: mean\nVertical line: min-max range',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Custom box plot 2: Total Energy
    ax2 = axes[1]
    for i, d in enumerate(durations):
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
    
    ax2.set_xlabel('Measurement Duration (seconds)', fontsize=12)
    ax2.set_ylabel('Total Energy (J)', fontsize=12)
    ax2.set_title('Distribution of Total Energy', fontsize=14)
    ax2.set_xticks(range(1, len(durations) + 1))
    ax2.set_xticklabels([str(d) for d in durations])
    ax2.set_xlim(0.5, len(durations) + 0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    # Add annotation
    ax2.text(0.02, 0.98, f'Box: mean ± std (n={num_repeats})\nRed line: mean\nVertical line: min-max range',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Energy Distribution by Measurement Duration', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(box_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Box plot saved to: {box_plot_path}")
    
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

