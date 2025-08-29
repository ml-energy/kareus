import json

with open("log/benchmark/component_scaling_efficiency/e2e_scaling_efficiency/e2e_scaling_efficiency_1.json", "r") as f:
    data_1 = json.load(f)

with open("log/benchmark/component_scaling_efficiency/e2e_scaling_efficiency/e2e_scaling_efficiency_2.json", "r") as f:
    data_2 = json.load(f)

with open("log/benchmark/component_scaling_efficiency/e2e_scaling_efficiency/e2e_scaling_efficiency_4.json", "r") as f:
    data_4 = json.load(f)

# with open("log/benchmark/component_scaling_efficiency/e2e_scaling_efficiency/e2e_scaling_efficiency_8.json", "r") as f:
#     data_8 = json.load(f)
data_8 = []

data = data_1 + data_2 + data_4 + data_8

with open("log/benchmark/component_scaling_efficiency/e2e_scaling_efficiency/e2e_scaling_efficiency.json", "w") as f:
    json.dump(data, f, indent=2)
