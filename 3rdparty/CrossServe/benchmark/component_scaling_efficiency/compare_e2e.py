import json

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

e2e_scaling_efficiency_path = (
    f"{LOG_DIR}/benchmark/component_scaling_efficiency/e2e_scaling_efficiency/e2e_scaling_efficiency.json"
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

try:
    with open(e2e_scaling_efficiency_path, "r") as f:
        e2e_scaling_efficiency = json.load(f)
except FileNotFoundError:
    e2e_scaling_efficiency = []

comm_dict = {}
for comm_item in comm_scaling_efficiency:
    key = (comm_item["bs"], comm_item["seq_len"], comm_item["ulysses_world_size"])
    comm_dict[key] = (
        comm_item["avg_uneven_time_scatter_idx_2_gather_idx_1"],
        comm_item["avg_uneven_time_scatter_idx_1_gather_idx_2"],
    )

mlp_dict = {}
for mlp_item in non_attn_scaling_efficiency:
    key = (mlp_item["bs"], mlp_item["seq_len"], mlp_item["ulysses_world_size"])
    mlp_dict[key] = (
        mlp_item["avg_time_multimodal_prologue"],
        mlp_item["avg_time_multimodal_epilogue"],
        mlp_item["avg_time_unimodal_prologue"],
        mlp_item["avg_time_unimodal_epilogue"],
    )

attn_dict = {}
for attn_item in ring_attn_scaling_efficiency:
    key = (attn_item["bs"], attn_item["seq_len"], attn_item["ulysses_world_size"], attn_item["ring_attn_world_size"])
    attn_dict[key] = attn_item["avg_time"]


e2e_dict = {}
for e2e_item in e2e_scaling_efficiency:
    key = (
        e2e_item["bs"],
        e2e_item["seq_len"],
        e2e_item["ulysses_degree"],
        e2e_item["ring_degree"],
        e2e_item["mlp_world_size"],
        e2e_item["steps"],
    )
    e2e_dict[key] = e2e_item["avg_e2e_time"]


for key in e2e_dict.keys():
    bs, seq_len, ulysses_degree, ring_degree, mlp_world_size, steps = key

    actual_time = e2e_dict[key]

    num_layers = 19
    num_single_layers = 38

    try:

        estimated_time = (
            (
                mlp_dict[(bs, seq_len, mlp_world_size)][0]
                + comm_dict[(bs, seq_len, ulysses_degree)][0]
                + attn_dict[(bs, seq_len, ulysses_degree, ring_degree)]
                + comm_dict[(bs, seq_len, ulysses_degree)][1]
                + mlp_dict[(bs, seq_len, mlp_world_size)][1]
            )
            * num_layers
            + (
                mlp_dict[(bs, seq_len, mlp_world_size)][2]
                + comm_dict[(bs, seq_len, ulysses_degree)][0]
                + attn_dict[(bs, seq_len, ulysses_degree, ring_degree)]
                + comm_dict[(bs, seq_len, ulysses_degree)][1]
                + mlp_dict[(bs, seq_len, mlp_world_size)][3]
            )
            * num_single_layers
        ) * steps

        diff_rate = abs(estimated_time - actual_time) / estimated_time
        print(
            f"bs: {bs}, seq_len: {seq_len}, ulysses_degree: {ulysses_degree}, ring_degree: {ring_degree}, mlp_world_size: {mlp_world_size}, steps: {steps}, estimated_time: {estimated_time:.2f} s, actual_time: {actual_time:.2f} s, diff_rate: {diff_rate * 100:.2f}%"
        )

    except KeyError:
        # print(f"KeyError: {key}")
        pass
