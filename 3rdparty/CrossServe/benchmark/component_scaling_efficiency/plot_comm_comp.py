import json
import matplotlib.pyplot as plt
import numpy as np

# LOG_DIR = "log_A100x4_80GB"
# LOG_DIR = "log_4xA40_48GB"
LOG_DIR = "log_8xH100_80GB"

comm_scaling_efficiency_path = (
    f"{LOG_DIR}/benchmark/component_scaling_efficiency/comm_scaling_efficiency/comm_scaling_efficiency.json"
)

try:
    ring_attn_scaling_efficiency_path = (
        f"{LOG_DIR}/benchmark/component_scaling_efficiency/ring_scaling_efficiency/ring_scaling_efficiency.json"
    )
except FileNotFoundError as e:
    print(e)
    ring_attn_scaling_efficiency_path = []

try:
    non_attn_scaling_efficiency_path = (
        f"{LOG_DIR}/benchmark/component_scaling_efficiency/non_attn_efficiency/non_attn_efficiency.json"
    )
except FileNotFoundError:
    non_attn_scaling_efficiency_path = None

with open(comm_scaling_efficiency_path, "r") as f:
    comm_scaling_efficiency = json.load(f)
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
comp_dict = {}

ring_attn_dict = {}
non_attn_dict = {}

for data in comm_scaling_efficiency:
    if data["mlp_world_size"] != data["attn_world_size"]:
        continue
    key = (data["bs"], data["seq_len"], data["hc"], data["hs"], data["mlp_world_size"])
    # comm_dict[key] = (data["avg_a2a_time_scatter_idx_2_gather_idx_1"], data["avg_a2a_time_scatter_idx_1_gather_idx_2"])
    comm_dict[key] = (data["avg_a2a_time_scatter_idx_2_gather_idx_1"] + data["avg_a2a_time_scatter_idx_1_gather_idx_2"]) * 2

for data in ring_attn_scaling_efficiency:
    if data["ring_attn_world_size"] != 1:
        continue
    key = (
        data["bs"],
        data["seq_len"],
        data["hc"],
        data["hs"],
        data["ulysses_world_size"],
    )
    ring_attn_dict[key] = data["avg_time"]
    comp_dict[key] = data["avg_time"]

for data in non_attn_scaling_efficiency:
    key = (data["bs"], data["seq_len"], data["hc"], data["hs"], data["ulysses_world_size"])
    non_attn_dict[key] = (
        data["avg_time_multimodal_prologue"],
        data["avg_time_unimodal_prologue"],
        data["avg_time_multimodal_epilogue"],
        data["avg_time_unimodal_epilogue"],
    )
    comp_dict[key] = comp_dict.get(key, 0) + sum(non_attn_dict[key])

# Prepare data for plotting
data_points = []
for key, comp_duration in comp_dict.items():
    bs, seq_len, hc, hs, mlp_world_size = key
    if mlp_world_size == 1:
        continue
    comm_duration = comm_dict[key]
    
    # Create label for x-axis
    # label = f"bs={bs}\nseq={seq_len}\nhc={hc}\nhs={hs}\nws={mlp_world_size}"
    # label = f"tokens={bs * seq_len}\nws={mlp_world_size}"
    label = f"{bs * seq_len} tokens, {mlp_world_size} gpu"
    total_time = comp_duration + comm_duration
    
    # Calculate sorting key
    sort_key = seq_len * bs * mlp_world_size
    
    data_points.append({
        'label': label,
        'sort_key': sort_key,
        'comp_pct': (comp_duration / total_time) * 100,
        'comm_pct': (comm_duration / total_time) * 100
    })

# Sort data points by seq_len * bs * world_size
data_points.sort(key=lambda x: x['sort_key'])

# Extract sorted data
labels = [dp['label'] for dp in data_points]
comp_percentages = [dp['comp_pct'] for dp in data_points]
comm_percentages = [dp['comm_pct'] for dp in data_points]

# Create the stacked bar chart
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(15, 8))
ax.bar(x, comp_percentages, width, label='Computation', color='skyblue')
ax.bar(x, comm_percentages, width, bottom=comp_percentages, label='Communication', color='lightcoral')

# Customize the plot
ax.set_ylabel('Percentage (%)')
ax.set_title('Computation vs Communication Time Distribution')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()

# Add percentage labels on the bars
for i in range(len(x)):
    comp_pct = comp_percentages[i]
    comm_pct = comm_percentages[i]
    
    # Add computation percentage label
    ax.text(i, comp_pct/2, f'{comp_pct:.1f}%', 
            ha='center', va='center')
    
    # Add communication percentage label
    ax.text(i, comp_pct + comm_pct/2, f'{comm_pct:.1f}%', 
            ha='center', va='center')

plt.tight_layout()
plt.savefig(f'{LOG_DIR}/comm_comp_distribution.png')
plt.close()


