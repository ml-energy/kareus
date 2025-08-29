import matplotlib.pyplot as plt
import json
import numpy as np
import itertools

# LOG_DIR = "log_A100x4_80GB"
# LOG_DIR = "log_4xA40_48GB"
# LOG_DIR = "log_8xH100_80GB"
LOG_DIR = "log"

with open(
    f"{LOG_DIR}/benchmark/component_scaling_efficiency/comm_scaling_efficiency/comm_scaling_efficiency.json", "r"
) as f:
    data = json.load(f)


def plot(scatter_idx, gather_idx):
    # Group data by batch_size and sequence length combination
    bs_seqlen_groups = {}
    for entry in data:
        key = (entry["bs"], entry["seq_len"])
        if key not in bs_seqlen_groups:
            bs_seqlen_groups[key] = []
        bs_seqlen_groups[key].append(entry)

    # Filter groups with at least 4 data points and sort by bs * seq_len
    filtered_groups = {k: v for k, v in bs_seqlen_groups.items() if len(v) >= 3}
    # sorted_groups = sorted(filtered_groups.items(), key=lambda x: x[0][0] * x[0][1])

    # Create subplots
    n_plots = len(filtered_groups)
    n_cols = 3  # You can adjust this for better layout
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axs = axs.flatten() if n_rows > 1 else [axs]

    # Sort by bs * seq_len product
    # sorted_groups = sorted(bs_seqlen_groups.items(), key=lambda x: x[0][0] * x[0][1])
    sorted_groups = sorted(filtered_groups.items(), key=lambda x: (x[0][1], x[0][0]))

    # Plot each bs-seqlen combination in a separate subplot

    for idx, ((bs, seq_len), entries) in enumerate(sorted_groups):
        # Sort entries first by mlp_world_size, then by attn_world_size
        entries = sorted(entries, key=lambda x: (x["mlp_world_size"], x["attn_world_size"]))

        world_sizes = [(entry["attn_world_size"], entry["mlp_world_size"]) for entry in entries]
        max_world_size = max(max(w) for w in world_sizes)

        xticks = list(range(len(world_sizes)))  # evenly spaced x-axis
        mlp_world_sizes = []
        for i, (attn_world_size, mlp_world_size) in enumerate(world_sizes):
            if attn_world_size == mlp_world_size:
                mlp_world_sizes.append(i)

        xticks_mlp = [tick for i, tick in enumerate(xticks) if world_sizes[i][1] == world_sizes[i][0]]
        uneven_times = [entry[f"uneven_time_scatter_idx_{scatter_idx}_gather_idx_{gather_idx}"] for entry in entries]
        # breakpoint()
        a2a_times = [
            entry[f"a2a_time_scatter_idx_{scatter_idx}_gather_idx_{gather_idx}"]
            for entry in entries
            if f"a2a_time_scatter_idx_{scatter_idx}_gather_idx_{gather_idx}" in entry
        ]

        # Plot measurements evenly spaced by (attn_world_size, mlp_world_size) on x-axis
        axs[idx].plot(xticks, uneven_times, "o-", label="Uneven All-to-All")
        axs[idx].plot(mlp_world_sizes, a2a_times, "o-", label="All-to-All")
        # set xticks to show as tuple of (attn_world_size, mlp_world_size)
        axs[idx].set_xticks(xticks)
        axs[idx].set_xticklabels([f"({w[0]}, {w[1]})" for w in world_sizes], rotation=45)

        axs[idx].set_title(f"Scatter{scatter_idx}, Gather{gather_idx}, BS={bs}, Seq_len={seq_len} (Total={bs*seq_len})")
        axs[idx].set_xlabel("(attn_world_size, mlp_world_size)")
        axs[idx].set_ylabel("Time (s)")
        axs[idx].grid(True)
        axs[idx].legend()

    # Remove empty subplots if any
    for idx in range(len(filtered_groups), len(axs)):
        fig.delaxes(axs[idx])

    plt.tight_layout()
    plt.savefig(
        f"{LOG_DIR}/benchmark/component_scaling_efficiency/comm_scaling_efficiency/comm_scaling_efficiency_scatter{scatter_idx}_gather{gather_idx}.png"
    )
    plt.close()


plot(2, 1)
plot(1, 2)
