import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

tp = 2
bs = 4
seq = 4096

freqs = [str(i) for i in range(1400, 900, -100)]

seq_times = []
seq_energies = []
ovlp_bestenergy_times = []
ovlp_bestenergy_energies = []

overlap_data = {}

for freq in freqs:
    df_base = pd.read_csv(f"logs/tp{tp}_bs{bs}_seq{seq}/{freq}/backward_energy_results_baseline.csv")
    df = pd.read_csv(f"logs/tp{tp}_bs{bs}_seq{seq}/{freq}/backward_energy_results.csv")

    row = df_base.iloc[0]
    # seq_times.append(row['0:time (s)'])
    # seq_energies.append(row['0:total energy (J)'])
    seq_time = row['0:time (s)']
    seq_energy = row['0:total energy (J)']

    overlap_rows = df[(df['overlap_start'] != -1) & 
                         (df['overlap_end'] != -1)]
    times = overlap_rows['time (s)'].values
    energies = overlap_rows['total energy (J)'].values

    idx_min_max_time = df['0:total energy (J)'].idxmin()
    min_row = df.loc[idx_min_max_time]
    # ovlp_bestenergy_times.append(min_row['0:time (s)'])
    # ovlp_bestenergy_energies.append(min_row['0:total energy (J)'])
    min_energy = min_row['0:total energy (J)']
    min_energy_time = min_row['0:time (s)']

    overlap_data[freq] = {
        'times': times,
        'energies': energies,
        'min_energy': min_energy,
        'min_energy_time': min_energy_time,
        'seq_energy': seq_energy,
        'seq_time': seq_time
    }

# Create the plot
plt.figure(figsize=(12, 9))

# Plot Perseus (no overlap) data
plt.scatter(seq_times, seq_energies, label='No overlap', alpha=0.7, s=80, color='blue')

# Define colormap for the frequency areas
colors = plt.cm.viridis(np.linspace(0, 1, len(freqs)))

# Plot overlap data for each frequency
for i, freq in enumerate(freqs):
    if freq in overlap_data:
        data = overlap_data[freq]
        
        # Find the convex hull of the points to create area
        if len(data['times']) >= 3:  # Need at least 3 points for convex hull
            from scipy.spatial import ConvexHull
            points = np.column_stack([data['times'], data['energies']])
            hull = ConvexHull(points)
            
            # Create polygon vertices from convex hull
            hull_points = points[hull.vertices]
            
            # Plot convex hull as polygon
            polygon = Polygon(hull_points, alpha=0.2, color=colors[i], label=f'Overlap with different configs')
            plt.gca().add_patch(polygon)
        else:
            # If not enough points, just plot the points
            plt.scatter(data['times'], data['energies'], alpha=0.2, color=colors[i])
        
        # Highlight minimum energy point
        plt.scatter(data['min_energy_time'], data['min_energy'], color=colors[i], 
                    s=100, marker='*', edgecolor='black', linewidth=1.5,
                    label=f'Min energy')
        
        # Add frequency label to minimum energy point
        plt.text(data['min_energy_time'], data['min_energy'], freq, 
                 fontsize=12, ha='right', va='bottom')

# Add frequency labels to Perseus points
for i, freq in enumerate(freqs):
    if i < len(seq_times):
        plt.text(seq_times[i], seq_energies[i], freq, 
                 fontsize=12, ha='right', va='bottom')

# Customize plot appearance
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel('Total Energy (J)', fontsize=20)
plt.title('Energy-Time Frontier of Overlap Configurations', fontsize=20)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Create a custom legend with fewer entries
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = []
unique_handles = []

# Filter to include only one entry for each category
seen_categories = set()
for handle, label in zip(handles, labels):
    category = label.split('(')[0].strip()
    if category not in seen_categories:
        seen_categories.add(category)
        unique_labels.append(label)
        unique_handles.append(handle)

plt.legend(unique_handles, unique_labels, fontsize=16, loc='best')

# Save the figure
plt.savefig("energy_time_regions.png", dpi=300, bbox_inches='tight')
plt.show() 