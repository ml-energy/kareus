import pandas as pd
import matplotlib.pyplot as plt

tp = 2
bs = 4
seq = 4096

freqs = [str(i) for i in range(1400, 900, -100)]

seq_times = []
seq_energies = []
ovlp_bestenergy_times = []
ovlp_bestenergy_energies = []

for freq in freqs:
    df_base = pd.read_csv(f"logs/tp{tp}_bs{bs}_seq{seq}/{freq}/energy_results_baseline.csv")
    df = pd.read_csv(f"logs/tp{tp}_bs{bs}_seq{seq}/{freq}/energy_results.csv")

    row = df_base.iloc[0]
    seq_times.append(row['0:time (s)'])
    seq_energies.append(row['0:total energy (J)'])

    idx_min_max_time = df['0:total energy (J)'].idxmin()
    min_row = df.loc[idx_min_max_time]
    ovlp_bestenergy_times.append(min_row['0:time (s)'] * 2)
    ovlp_bestenergy_energies.append(min_row['0:total energy (J)'] * 2)

plt.figure(figsize=(10, 8))
plt.scatter(seq_times, seq_energies, label='Perseus (no overlap)', alpha=0.7, s=80)
plt.scatter(ovlp_bestenergy_times, ovlp_bestenergy_energies, label='best energy in current frequency', alpha=0.7, s=80)

for i, label in enumerate(freqs):
    x = seq_times[i]
    y = seq_energies[i]
    # You can adjust ha, va, and/or add offsets to avoid label-marker overlap
    plt.text(x, y, label, fontsize=12, ha='right', va='bottom')

    m = ovlp_bestenergy_times[i]
    n = ovlp_bestenergy_energies[i]
    plt.text(m, n, label, fontsize=12, ha='right', va='bottom')

plt.xlabel('time (s)', fontsize=20)
plt.ylabel('total energy (J)', fontsize=20)
plt.title(f'time vs. energy in different frequencies', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig("frontier_ovlp.png", dpi=300)
