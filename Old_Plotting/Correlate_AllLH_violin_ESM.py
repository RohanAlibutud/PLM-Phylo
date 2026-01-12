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

# === Define base folders ===
base_esm_folder = "/home/rohan/ESM/Results/OMAM20/OMAM20_34spp/Attention_Matrices"
base_pdist_folder = "/home/rohan/ESM/Results/MEGA_Pdist_Matrices/"

# Dictionary to store correlation values and significance count
protein_correlations = {protein: [] for protein in protein_list}
significant_counts = {protein: 0 for protein in protein_list}
total_layer_heads = {protein: 0 for protein in protein_list}
protein_pvalues = {protein: [] for protein in protein_list}


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

    # === Load ESM Attention Distance Matrices ===
    npz_file = f"{base_esm_folder}/{protein}/{protein}_distance_matrices.npz"
    if not os.path.exists(npz_file):
        print(f"Skipping {protein}: ESM matrix file not found.")
        continue

    data = np.load(npz_file)["arr_0"]  # Extract array
    num_layers, num_heads, num_species, _ = data.shape

    # === Load and Convert P-Distance Matrix ===
    p_distance_file = f"{base_pdist_folder}/{protein}_34sppAligned_pdist.csv"
    if not os.path.exists(p_distance_file):
        print(f"Skipping {protein}: P-Distance file not found.")
        continue

    p_distance_matrix = load_lower_triangular_with_labels(p_distance_file)
    p_distance_flat = p_distance_matrix[np.triu_indices_from(p_distance_matrix, k=1)]

    # === Compute Correlations for Each Layer and Head (No Mean) ===
    for layer in range(num_layers):
        for head in range(num_heads):
            attention_matrix = data[layer, head]
            attention_flat = attention_matrix[np.triu_indices_from(attention_matrix, k=1)]

            if np.all(attention_flat == attention_flat[0]):  # Skip constant values
                continue

            pearson_corr, p_value = stats.pearsonr(p_distance_flat, attention_flat)
            protein_correlations[protein].append(pearson_corr)  # Store RAW correlation value
            #protein_correlations[protein].append(abs(pearson_corr))  # Store ABSOLUTE correlation value

            # Track total layer-heads and significant ones
            total_layer_heads[protein] += 1
            if p_value < 0.05:
                significant_counts[protein] += 1

    protein_correlations[protein].append(pearson_corr)
    protein_pvalues[protein].append(p_value)


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
    pc.set_facecolor('dodgerblue')
    pc.set_alpha(0.6)

# Set x-axis labels and title
plt.xticks(ticks=np.arange(len(valid_proteins)), labels=valid_proteins, rotation=60, fontsize=14)
plt.ylabel("Pearson Correlation", fontsize=18)
plt.title("ESM-2 attention distance vs. p-distance over all layerheads", fontsize=20)
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
"""

# Show Plot
plt.show()

# SIGNIFICANCE PIE CHART
total_significant_heads = 0
for protein in protein_list:
    total_significant_heads += significant_counts[protein]
print(f"Total count of significant heads: {total_significant_heads}")
total_heads = 660 * len(protein_list)
total_heads_significant_percentage = total_significant_heads / total_heads
print(f"Total percentage of heads significant: {total_heads_significant_percentage}")

labels = ["Significant", "Not Significant"]
sizes = [total_significant_heads, total_heads - total_significant_heads]
fig, ax = plt.subplots(figsize=(6, 6))
colors = ["limegreen", "darkgreen"]
explode = (0.05, 0.05)
ax.set_title("Percentage of heads with\nsignificant correlations, ESM-2", fontsize=26)

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    explode=explode,
    textprops={'fontsize': 18},
    autopct='%1.1f%%',
    startangle=0
)

# FIRST wedge styling
autotexts[0].set_color('black')

# SECOND wedge — calculate position offset:
wedge2 = wedges[1]
angle2 = (wedge2.theta2 + wedge2.theta1) / 2
x2 = np.cos(np.radians(angle2))
y2 = np.sin(np.radians(angle2))

# Move second wedge's number outward
autotexts[1].set_color('black')
autotexts[1].set_position((.7 * x2, -1.5 * y2))  # adjust factor as needed

plt.show()



# === GLOBAL STATISTICS ===
# Flatten all correlations and p-values
all_correlations = [r for protein in protein_list for r in protein_correlations[protein]]
all_pvalues = [p for protein in protein_list for p in protein_pvalues[protein]]

# Mean and Median Correlation
mean_corr = np.mean(all_correlations)
median_corr = np.median(all_correlations)
mean_pval = np.mean(all_pvalues)

# Wilcoxon Signed-Rank Test: H0 = median correlation == 0
try:
    w_stat, wilcoxon_p = stats.wilcoxon(all_correlations)
except ValueError as e:
    w_stat, wilcoxon_p = None, None
    print("Wilcoxon test could not be performed:", e)

# === Print Global Summary ===
print("\n=== GLOBAL SUMMARY STATISTICS ===")
print(f"Mean Pearson correlation across all layer-heads: {mean_corr:.4f}")
print(f"Median Pearson correlation: {median_corr:.4f}")
print(f"Mean p-value across all layer-heads: {mean_pval:.4e}")
if wilcoxon_p is not None:
    print(f"Wilcoxon signed-rank test p-value (H0: median == 0): {wilcoxon_p:.4e}")
else:
    print("Wilcoxon test not computed due to constant input or insufficient data.")