import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# === Define the list of proteins ===
protein_list = [
    "AIMP2", "APOBEC2", "AR", "CASQ1", 
    "COCH", "DYRK1B", "GPX3", "HAUS1", 
    "IL6", "INO80B", "KRT23", "NEUROG1",
    "OPN1SW", "PPOX", "RHO", "SLC39A12", 
    "TF", "TPPP2", "UBA2", "UPRT"
]

# === Define file paths ===
base_msat_folder = "/home/rohan/ESM/Results/OMAM20/MSAT/OMAM20_34sppAligned/Attention_Matrices"
base_pdist_folder = "/home/rohan/ESM/Results/MEGA_Pdist_Matrices/"

# Dictionary to store correlation values and significance count
protein_correlations = {protein: [] for protein in protein_list}
significant_counts = {protein: 0 for protein in protein_list}
total_layer_heads = {protein: 0 for protein in protein_list}

# === Function to Load Lower Triangular P-Distance Matrix ===
def load_lower_triangular_with_labels(file_path):
    df = pd.read_csv(file_path)
    species = df.iloc[:, 0].values  # Extract species names
    df_values = df.iloc[:, 1:].values.astype(float)  # Convert values to float matrix
    size = len(species)
    full_matrix = np.zeros((size, size))  # Create square matrix

    for i in range(size):
        full_matrix[i, :i+1] = df_values[i, :i+1]
        full_matrix[:i+1, i] = df_values[i, :i+1]  # Mirror upper triangle

    return full_matrix

# === Loop over each protein ===
for protein in protein_list:
    print(f"\nProcessing {protein}...")

    # === Load MSAT Attention Distance Matrices ===
    mean_matrices_folder = f"{base_msat_folder}/{protein}/{protein}_Matrices/Meanmatrices"
    if not os.path.exists(mean_matrices_folder):
        print(f"Skipping {protein}: MSAT matrix folder not found.")
        continue

    # === Load and Convert P-Distance Matrix ===
    p_distance_file = f"{base_pdist_folder}/{protein}_34sppAligned_pdist.csv"
    if not os.path.exists(p_distance_file):
        print(f"Skipping {protein}: P-Distance file not found.")
        continue

    p_distance_matrix = load_lower_triangular_with_labels(p_distance_file)
    p_distance_flat = p_distance_matrix[np.triu_indices_from(p_distance_matrix, k=1)]

    # === Compute Correlations for Each Layer and Head ===
    for layer in range(1, 13):
        for head in range(1, 13):
            attention_file = f"{mean_matrices_folder}/{protein}_layer{layer}_head{head}_meanmatrix.csv"
            
            if not os.path.exists(attention_file):
                continue

            att_matrix = pd.read_csv(attention_file, header=None).values.astype(float)
            
            # Find max species attention in this layer-head matrix for normalization
            max_att = np.max(att_matrix)
            
            # Compute attention distance
            aDist_matrix = max_att - att_matrix
            aDist_flat = aDist_matrix[np.triu_indices_from(aDist_matrix, k=1)]  # Exclude diagonal

            pearson_corr, p_value = stats.pearsonr(p_distance_flat, aDist_flat)
            protein_correlations[protein].append(pearson_corr)  # Store RAW correlation value
            #protein_correlations[protein].append(abs(pearson_corr))  # Store ABSOLUTE correlation value

            # Track total layer-heads and significant ones
            total_layer_heads[protein] += 1
            if p_value < 0.05:
                significant_counts[protein] += 1

# === Calculate Percentage of Significant Layer-Heads ===
significance_percentages = {
    protein: (significant_counts[protein] / total_layer_heads[protein] * 100) if total_layer_heads[protein] > 0 else 0
    for protein in protein_list
}

# === Print Results ===
for protein in protein_list:
    if total_layer_heads[protein] > 0:
        print(f"{protein}: {significant_counts[protein]}/{total_layer_heads[protein]} "
              f"layer-heads significant ({significance_percentages[protein]:.2f}%)")

# === Prepare Data for Violin Plot ===
valid_proteins = [protein for protein in protein_list if protein_correlations[protein]]
violin_data = [protein_correlations[protein] for protein in valid_proteins]

# === Generate Violin Plot ===
plt.figure(figsize=(12, 6))

# Violin Plot for Full Distribution of Correlations Across Layers and Heads
vp = plt.violinplot(violin_data, positions=np.arange(len(valid_proteins)), showmeans=True, showmedians=True, widths=0.5)
for pc in vp['bodies']:
    pc.set_facecolor('orange')
    pc.set_alpha(0.6)
for partname in ('cmeans', 'cmedians', 'cbars'):  
    vp[partname].set_color('darkorange')  

# Set x-axis labels and title
plt.xticks(ticks=np.arange(len(valid_proteins)), labels=valid_proteins, rotation=60, fontsize=14)
plt.ylabel("Pearson Correlation", fontsize=18)
plt.title("MSAT attention distance vs. p-distance over all layerheads", fontsize=20)
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)  # Baseline at 0
plt.grid(axis="y", linestyle="--", alpha=0.7)

"""
# === Annotate with % of significant layer-heads ===
for i, protein in enumerate(valid_proteins):
    plt.annotate(
        f"{significance_percentages[protein]:.1f}%",
        xy=(i, max(protein_correlations[protein])),
        xytext=(0, 8), 
        textcoords='offset points',
        fontsize=12,
        ha='center',
        color='black',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.7)
    )

# Show Plot
"""

plt.show()


# SIGNIFICANCE PIE CHART
total_significant_heads = 0
for protein in protein_list:
    #print(f"{protein}: {significant_counts[protein]}")
    total_significant_heads += significant_counts[protein]
print(f"Total count of significant heads: {total_significant_heads}")
total_heads = 144 * len(protein_list)
total_heads_significant_percentage = total_significant_heads/total_heads
print(f"Total percentage of heads significant: {total_heads_significant_percentage}")

labels = ["Significant", "Not Significant"]
sizes = [total_significant_heads, total_heads - total_significant_heads]
fig, ax = plt.subplots(figsize = (6, 6))
colors = ["limegreen", "darkgreen"]
explode = (0.05, 0.05)
ax.set_title("Percentage of heads with\nsignificant correlations, MSA-PLM", fontsize = 26)
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors = colors, explode = explode, textprops={'fontsize': 18}, autopct='%1.1f%%')
autotexts[0].set_color('black')  # first wedge
autotexts[1].set_color('white')
