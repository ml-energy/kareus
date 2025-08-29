import json
import matplotlib.pyplot as plt
import numpy as np

# Read the JSON data
with open("log/benchmark/component_scaling_efficiency/cost_simulator.json", "r") as f:
    data = json.load(f)

# Define colors for each cost type
colors = {"naive": "#FF6B6B", "scaling": "#4ECDC4", "disaggregated": "#45B7D1"}  # Red  # Turquoise  # Blue

# Create a figure with a grid of subplots
n_entries = len(data)
n_cols = 4
n_rows = (n_entries + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
axes = axes.flatten()

# Plot each entry
for idx, entry in enumerate(data):
    ax = axes[idx]

    # Data for plotting
    costs = [entry["naive_cost"], entry["scaling_cost"], entry["disaggregated_cost"]]
    labels = ["Naive", "Scaling", "Disaggregated"]

    # Create bars with colors
    x = np.arange(len(labels))
    bars = ax.bar(x, costs, color=[colors["naive"], colors["scaling"], colors["disaggregated"]])

    # Customize the subplot
    ax.set_ylabel("Cost")
    ax.set_title(f'bs={entry["bs"]}, seq_len={entry["seq_len"]}\nhc={entry["hc"]}, hs={entry["hs"]}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom")

# Remove empty subplots if any
for idx in range(len(data), len(axes)):
    fig.delaxes(axes[idx])

# Add a legend to the figure
fig.legend(bars, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)

# Adjust layout
plt.tight_layout()
# plt.show()
plt.savefig("log/benchmark/component_scaling_efficiency/cost_simulator.png")
