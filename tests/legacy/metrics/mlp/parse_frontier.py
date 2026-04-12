import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

tp = 2
bs = 8
seq = 4096

freqs = [str(i) for i in range(1400, 900, -100)]

seq_times = []
seq_total_energies = []
seq_effective_energies = []

# Best (min) points per frequency
ovlp_best_total_times = []
ovlp_best_total_energies = []
ovlp_best_effective_times = []
ovlp_best_effective_energies = []

# Overlap point clouds per frequency for convex hulls
overlap_data_total = {}
overlap_data_effective = {}

for freq in freqs:
    df_base = pd.read_csv(f"logs/tp{tp}-bs{bs}-seq{seq}/{freq}/energy_results_baseline.csv")
    df = pd.read_csv(f"logs/tp{tp}-bs{bs}-seq{seq}/{freq}/energy_results.csv")

    row = df_base.iloc[0]
    seq_times.append(row['0:time (s)'])
    seq_total_energies.append(row['0:total energy (J)'])
    # Baseline effective energy uses the same formula
    seq_effective_energies.append(row['0:total energy (J)'] - 70 * row['0:time (s)'])

    idx_min_total_energy = df['0:total energy (J)'].idxmin()
    df['effe_energy'] = df['0:total energy (J)'] - 70 * df['0:time (s)'] * 2
    idx_min_effective_energy = df['effe_energy'].idxmin()
    
    # Best total energy point (scaled to match overlap convention)
    min_total_row = df.loc[idx_min_total_energy]
    ovlp_best_total_times.append(min_total_row['0:time (s)'] * 2)
    ovlp_best_total_energies.append(min_total_row['0:total energy (J)'] * 2)

    # Best effective energy point (time scaled; energy already effective)
    min_effective_row = df.loc[idx_min_effective_energy]
    ovlp_best_effective_times.append(min_effective_row['0:time (s)'] * 2)
    ovlp_best_effective_energies.append(min_effective_row['effe_energy'] * 2)

    # Collect all overlap configurations for convex hull (scale by 2 to match min-energy points)
    times = (df['0:time (s)'] * 2).values
    energies_total = (df['0:total energy (J)'] * 2).values
    energies_effective = (df['effe_energy'] * 2).values
    if len(times) > 0:
        overlap_data_total[freq] = {
            'times': times,
            'energies': energies_total,
            'min_energy_time': ovlp_best_total_times[-1],
            'min_energy': ovlp_best_total_energies[-1],
        }
        overlap_data_effective[freq] = {
            'times': times,
            'energies': energies_effective,
            'min_energy_time': ovlp_best_effective_times[-1],
            'min_energy': ovlp_best_effective_energies[-1],
        }

# Common colors for frequencies
colors = plt.cm.viridis(np.linspace(0, 1, len(freqs)))

# ----- Total Energy figure -----
fig_total, ax = plt.subplots(figsize=(12, 8))
ax.scatter(seq_times, seq_total_energies, label='No overlap', alpha=0.7, s=80, color='tab:blue')
hull_label_plotted = False
star_label_plotted = False
for i, freq in enumerate(freqs):
    if freq in overlap_data_total:
        data = overlap_data_total[freq]
        points = np.column_stack([data['times'], data['energies']])
        if points.shape[0] >= 3:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            polygon = Polygon(
                hull_points,
                alpha=0.2,
                color=colors[i],
                # label='Overlap convex hull' if not hull_label_plotted else None,
            )
            ax.add_patch(polygon)
            hull_label_plotted = True
        else:
            ax.scatter(
                data['times'],
                data['energies'],
                alpha=0.2,
                color=colors[i],
                label='Overlap points' if not hull_label_plotted else None,
            )
            hull_label_plotted = True

        ax.scatter(
            data['min_energy_time'],
            data['min_energy'],
            color=colors[i],
            s=100,
            marker='*',
            edgecolor='black',
            linewidth=1.5,
            label='Min total energy' if not star_label_plotted else None,
        )
        star_label_plotted = True

for i, label in enumerate(freqs):
    if i < len(seq_times):
        x = seq_times[i]
        y = seq_total_energies[i]
        ax.text(x, y, label, fontsize=12, ha='right', va='bottom')
    if i < len(ovlp_best_total_times):
        m = ovlp_best_total_times[i]
        n = ovlp_best_total_energies[i]
        ax.text(m, n, label, fontsize=12, ha='right', va='bottom')

ax.set_xlabel('time (s)', fontsize=16)
ax.set_ylabel('total energy (J)', fontsize=16)
# ax.set_title('Total energy vs time', fontsize=18)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
fig_total.tight_layout()
fig_total.savefig("frontier_ovlp_total.pdf", dpi=300)

# ----- Effective Energy figure -----
fig_eff, ax = plt.subplots(figsize=(12, 8))
ax.scatter(seq_times, seq_effective_energies, label='No overlap', alpha=0.7, s=80, color='tab:orange')
hull_label_plotted = False
star_label_plotted = False
for i, freq in enumerate(freqs):
    if freq in overlap_data_effective:
        data = overlap_data_effective[freq]
        points = np.column_stack([data['times'], data['energies']])
        if points.shape[0] >= 3:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            polygon = Polygon(
                hull_points,
                alpha=0.2,
                color=colors[i],
                # label='Overlap convex hull' if not hull_label_plotted else None,
            )
            ax.add_patch(polygon)
            hull_label_plotted = True
        else:
            ax.scatter(
                data['times'],
                data['energies'],
                alpha=0.2,
                color=colors[i],
                label='Overlap points' if not hull_label_plotted else None,
            )
            hull_label_plotted = True

        ax.scatter(
            data['min_energy_time'],
            data['min_energy'],
            color=colors[i],
            s=100,
            marker='*',
            edgecolor='black',
            linewidth=1.5,
            label='Min effective energy' if not star_label_plotted else None,
        )
        star_label_plotted = True

for i, label in enumerate(freqs):
    if i < len(seq_times):
        x = seq_times[i]
        y = seq_effective_energies[i]
        ax.text(x, y, label, fontsize=12, ha='right', va='bottom')
    if i < len(ovlp_best_effective_times):
        m = ovlp_best_effective_times[i]
        n = ovlp_best_effective_energies[i]
        ax.text(m, n, label, fontsize=12, ha='right', va='bottom')

ax.set_xlabel('time (s)', fontsize=16)
ax.set_ylabel('effective energy (J)', fontsize=16)
# ax.set_title('Effective energy vs time', fontsize=18)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
fig_eff.tight_layout()
fig_eff.savefig("frontier_ovlp_effective.pdf", dpi=300)
