import matplotlib.pyplot as plt
import json
import numpy as np

LOG_DIR = "log_A100x4_80GB"
# LOG_DIR = "log_4xA40_48GB"
# LOG_DIR = "log_8xH100_80GB"
# LOG_DIR = "log"


with open(f"{LOG_DIR}/benchmark/component_scaling_efficiency/non_attn_efficiency/non_attn_efficiency.json", "r") as f:
    data = json.load(f)


def plot_non_attn_efficiency(part_name: str = "multimodal_prologue"):
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
        # Sort entries by ulysses_world_size
        entries = sorted(entries, key=lambda x: x["ulysses_world_size"])

        ulysses_world_sizes = [entry["ulysses_world_size"] for entry in entries]
        times = [entry[f"avg_time_{part_name}"] for entry in entries]

        # Plot actual measurements
        axs[idx].plot(ulysses_world_sizes, times, "o-", label="Actual")

        # Plot ideal scaling reference line
        if len(ulysses_world_sizes) > 0 and len(times) > 0:
            base_ulysses_world_size = ulysses_world_sizes[0]
            base_time = times[0]
            ideal_times = [base_time / (w_size / base_ulysses_world_size) for w_size in ulysses_world_sizes]
            axs[idx].plot(ulysses_world_sizes, ideal_times, "--", color="orange", label="Ideal Scaling")

        axs[idx].set_title(f"BS={bs}, Seq_len={seq_len} (Total={bs*seq_len})")
        axs[idx].set_xlabel("World Size")
        axs[idx].set_ylabel("Time (s)")
        axs[idx].grid(True)
        axs[idx].legend()

    # Remove empty subplots if any
    for idx in range(len(filtered_groups), len(axs)):
        fig.delaxes(axs[idx])

    plt.tight_layout()
    plt.savefig(
        f"{LOG_DIR}/benchmark/component_scaling_efficiency/non_attn_efficiency/non_attn_efficiency_{part_name}.png"
    )
    plt.close()


if __name__ == "__main__":
    plot_non_attn_efficiency("multimodal_prologue")
    plot_non_attn_efficiency("unimodal_prologue")
    plot_non_attn_efficiency("multimodal_epilogue")
    plot_non_attn_efficiency("unimodal_epilogue")
