import pandas as pd
import glob
import re
import sys
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

# freqs = [str(i) for i in range(1700, 900, -100)]
freqs = ['1400']
tensor_parallel = 2
batch_size = 4
seq_len = 4096

plt.figure(figsize=(10, 6))

for freq in freqs:
    data = pd.read_csv(f"logs/tp{tensor_parallel}-bs{batch_size}-seq{seq_len}/{freq}/energy_results.csv")
    # data = data[data['0:total energy (J)'] < 16]
    data = data[data['0:time (s)'] < 0.003]
    plt.scatter(data['0:time (s)'], data['0:total energy (J)'], label=freq, alpha=0.7)

plt.title("attention forward with different overlap scopes and SM configs", fontsize=14)
plt.xlabel("time (s)", fontsize=12)
plt.ylabel("energy (J)", fontsize=12)
plt.legend()
plt.grid(alpha=0.5)

plt.savefig("attn_forward.png")


# if __name__ == "__main__":
#     freq = sys.argv[1]
#     repeat = 3
#     tensor_parallel = 2
#     batch_size = 4
#     seq_len = 4096

#     df_overlap = pd.read_csv(f'logs/tp{tensor_parallel}-bs{batch_size}-seq{seq_len}/{freq}/backward_energy_results.csv')
#     print(df_overlap.head())

#     long_data = []
#     for i in range(repeat):
#         time_col = f"{i}:time (s)"
#         energy_col = f"{i}:total energy (J)"
#         temp_df = df_overlap[['overlap_start', 'overlap_end', 'comm_sm_number', 'comm_block_size']].copy()
#         temp_df['iteration'] = i
#         temp_df['time'] = df_overlap[time_col]
#         temp_df['avg_energy'] = df_overlap[energy_col] / tensor_parallel
#         long_data.append(temp_df)

#     df_overlap_long = pd.concat(long_data, ignore_index=True)
#     print(df_overlap_long.head())

    