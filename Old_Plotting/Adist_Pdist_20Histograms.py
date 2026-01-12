"""
Fig. B.1.2.2
-------------------------------------------------------------------------------
Histograms of the # of attention heads that achieved a given correlation 
between ESM attention distance and p-distance, for each of seven proteins. 
"""


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
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
base_pdist_folder = "/home/rohan/ESM/Results/MEGA_Pdist_Matrices/34sppAligned/"

# === Function to Load Lower Triangular P-Distance Matrix ===
def load_lower_triangular_with_labels(file_path):
    """ Reads a labeled lower triangular distance matrix from a CSV file and converts it to a full symmetric matrix. """
    df = pd.read_csv(file_path)
    species = df.iloc[:, 0].values  # Extract species names
    df_values = df.iloc[:, 1:].values.astype(float)  # Convert values to float matrix
    size = len(species)
    full_matrix = np.zeros((size, size))  # Create square matrix

    # Fill lower triangle from CSV
    for i in range(size):
        full_matrix[i, :i+1] = df_values[i, :i+1]
        full_matrix[:i+1, i] = df_values[i, :i+1]  # Mirror upper triangle

    return full_matrix

# Prepare the figure for 7 subplots in a 2x3 grid plus 1 extra plot
fig, axes = plt.subplots(5, 4, figsize=(15, 10))
axes = axes.flatten()

for idx, protein in enumerate(protein_list):
    print(f"\nProcessing {protein}...")

    # === Define File Paths ===
    npz_file = os.path.join(base_esm_folder, f"{protein}/{protein}_distance_matrices.npz")
    pdist_file = os.path.join(base_pdist_folder, f"{protein}_34sppAligned_pdist.csv")

    # === STEP 1: Load Attention Distance Matrices ===
    try:
        data = np.load(npz_file)["arr_0"]
    except FileNotFoundError:
        print(f"Warning: NPZ file not found for {protein}, skipping.")
        continue

    assert data.shape == (33, 20, 34, 34), f"Unexpected shape {data.shape} for {protein}, expected (33, 20, 34, 34)"

    # === STEP 2: Load and Convert the Lower Triangular P-Distance Matrix ===
    try:
        p_distance_matrix = load_lower_triangular_with_labels(pdist_file)
    except FileNotFoundError:
        print(f"Warning: P-Distance file not found for {protein}, skipping.")
        continue

    assert p_distance_matrix.shape == (34, 34), f"Unexpected p-distance matrix shape: {p_distance_matrix.shape} for {protein}"

    # Flatten upper triangle for correlation (excluding diagonal)
    p_distance_flat = p_distance_matrix[np.triu_indices_from(p_distance_matrix, k=1)]

    # === STEP 3: Compute Correlation Across Heads ===
    num_layers, num_heads, num_species, _ = data.shape
    assert num_species == 34, "Mismatch in species count between matrices"

    head_correlations = []  # Store Pearson correlations for each head across all layers
    max_correlation = -np.inf
    best_layer, best_head = None, None
    significant_count = 0
    total_heads = num_layers * num_heads

    for layer in tqdm(range(num_layers), desc=f"Processing Layers for {protein}"):
        for head in range(num_heads):
            attention_matrix = data[layer, head]

            # Flatten upper triangular (excluding diagonal)
            attention_flat = attention_matrix[np.triu_indices_from(attention_matrix, k=1)]

            # Check if values are valid for correlation
            if np.all(attention_flat == attention_flat[0]):  # Check for constant values
                pearson_corr = np.nan
                p_value = np.nan
            else:
                pearson_corr, p_value = stats.pearsonr(p_distance_flat, attention_flat)

            head_correlations.append(pearson_corr)

            # Track highest correlation
            if not np.isnan(pearson_corr) and pearson_corr > max_correlation:
                max_correlation = pearson_corr
                best_layer, best_head = layer, head

            # Count significant correlations (p < 0.05)
            if not np.isnan(p_value) and p_value < 0.05:
                significant_count += 1

    # Compute percentage of significant correlations
    significant_percentage = (significant_count / total_heads) * 100

    # === STEP 4: Plot Histogram of Head-wise Pearson Correlations ===
    ax = axes[idx]  # Select the appropriate subplot
    ax.hist(head_correlations, bins=20, color='forestgreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel("Pearson Correlation between Adist and Pdist")
    ax.set_ylabel("# of Attention Heads")
    ax.set_title(f"{protein}")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)  # Baseline at 0
    ax.grid(True)
    ax.text(-0.10, 1, str(idx + 1), transform=ax.transAxes, fontsize=16, fontweight='bold', color='black')



    # Print results
    print(f"Highest Pearson Correlation for {protein}: {max_correlation:.4f} at Layer {best_layer}, Head {best_head}")
    print(f"Percentage of Significant Correlations (p < 0.05) for {protein}: {significant_percentage:.2f}%")

# Adjust layout and show plots
plt.tight_layout()
plt.show()
