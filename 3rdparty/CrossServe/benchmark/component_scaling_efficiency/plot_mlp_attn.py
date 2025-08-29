import json
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = "log_A100x4_80GB"
# LOG_DIR = "log_4xA40_48GB"
# LOG_DIR = "log_8xH100_80GB"

comm_scaling_efficiency_path = (
    f"{LOG_DIR}/benchmark/component_scaling_efficiency/comm_scaling_efficiency/comm_scaling_efficiency.json"
)

ring_attn_scaling_efficiency_path = (
    f"{LOG_DIR}/benchmark/component_scaling_efficiency/ring_scaling_efficiency/ring_scaling_efficiency.json"
)

non_attn_scaling_efficiency_path = (
    f"{LOG_DIR}/benchmark/component_scaling_efficiency/non_attn_efficiency/non_attn_efficiency.json"
)

try:
    with open(comm_scaling_efficiency_path, "r") as f:
        comm_scaling_efficiency = json.load(f)
except FileNotFoundError:
    comm_scaling_efficiency = []

try:
    with open(ring_attn_scaling_efficiency_path, "r") as f:
        ring_attn_scaling_efficiency = json.load(f)
except FileNotFoundError:
    ring_attn_scaling_efficiency = []

try:
    with open(non_attn_scaling_efficiency_path, "r") as f:
        non_attn_scaling_efficiency = json.load(f)
except FileNotFoundError:
    non_attn_scaling_efficiency = []

comm_dict = {}
for comm_item in comm_scaling_efficiency:
    key = (comm_item["bs"], comm_item["seq_len"], comm_item["ulysses_world_size"])
    comm_dict[key] = (
        comm_item["avg_uneven_time_scatter_idx_2_gather_idx_1"],
        comm_item["avg_uneven_time_scatter_idx_1_gather_idx_2"],
    )

# Prepare data for plotting
data_points = []
for item in non_attn_scaling_efficiency:
    bs = item["bs"]
    seq_len = item["seq_len"]
    gpu_num = item["ulysses_world_size"]

    mlp_time = (
        item["avg_time_unimodal_prologue"]
        + item["avg_time_unimodal_epilogue"]
        + item["avg_time_multimodal_prologue"]
        + item["avg_time_multimodal_epilogue"]
    )

    attn_time = float("inf")

    for attn_item in ring_attn_scaling_efficiency:
        if (
            attn_item["bs"] == bs
            and attn_item["seq_len"] == seq_len
            and attn_item["ulysses_world_size"] * attn_item["ring_attn_world_size"] == gpu_num
        ):
            comm_time = comm_dict.get((bs, seq_len, gpu_num), (0, 0))
            comm_time = sum(comm_time)
            attn_time = min(attn_time, attn_item["avg_time"] + comm_time)

    attn_time = attn_time * 2

    total_time = mlp_time + attn_time
    mlp_pct = (mlp_time / total_time) * 100
    attn_pct = (attn_time / total_time) * 100

    # Create label for x-axis
    label = f"{bs}x{seq_len} tokens, {gpu_num} gpu"

    # Calculate sorting key
    sort_key = seq_len * bs * gpu_num

    data_points.append({"label": label, "sort_key": sort_key, "attn_pct": attn_pct, "mlp_pct": mlp_pct})

# Sort data points by seq_len * bs * world_size
data_points.sort(key=lambda x: x["sort_key"])

# Group data points by GPU count
gpu_groups = {}
for dp in data_points:
    gpu_num = int(dp["label"].split(", ")[1].split(" ")[0])  # Extract GPU number
    if gpu_num not in gpu_groups:
        gpu_groups[gpu_num] = []
    gpu_groups[gpu_num].append(dp)

# Sort GPU numbers
gpu_nums = sorted(gpu_groups.keys())

# Create subplots
n_plots = len(gpu_nums)
fig, axes = plt.subplots(n_plots, 1, figsize=(15, 6 * n_plots))
if n_plots == 1:
    axes = [axes]  # Make axes iterable when there's only one subplot

for ax, gpu_num in zip(axes, gpu_nums):
    # Sort data points for this GPU count
    gpu_data = sorted(gpu_groups[gpu_num], key=lambda x: x["sort_key"])

    # Extract data for this GPU count
    labels = [dp["label"].split(", ")[0] for dp in gpu_data]  # Only show tokens part
    attn_percentages = [dp["attn_pct"] for dp in gpu_data]
    mlp_percentages = [dp["mlp_pct"] for dp in gpu_data]

    # Create the stacked bar chart
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x, attn_percentages, width, label="Attention", color="skyblue")
    ax.bar(x, mlp_percentages, width, bottom=attn_percentages, label="MLP", color="lightcoral")

    # Customize the subplot
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Attention vs MLP Time Distribution ({gpu_num} GPUs)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    # Add percentage labels on the bars
    for i in range(len(x)):
        attn_pct = attn_percentages[i]
        mlp_pct = mlp_percentages[i]

        # Add MLP percentage label
        ax.text(i, attn_pct + mlp_pct / 2, f"{mlp_pct:.1f}%", ha="center", va="center")

        # Add attention percentage label
        ax.text(i, attn_pct / 2, f"{attn_pct:.1f}%", ha="center", va="center")

plt.tight_layout()
plt.savefig(f"{LOG_DIR}/mlp_attn_distribution.png")
plt.close()
