"""
Fig. B.1.2.3a | Mean correlation barchart
-------------------------------------------------------------------------------
Correlation with p-distance for ESM-2 and MSA Transformer attention distance 
from the Mean correlation attention head. 
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
base_msat_folder = "/home/rohan/ESM/Results/OMAM20/MSAT/OMAM20_34sppAligned/Attention_Matrices"
base_pdist_folder = "/home/rohan/ESM/Results/MEGA_Pdist_Matrices/34sppAligned"


# Store mean correlations for each protein
esm_mean_correlations = []
msat_mean_correlations = []

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
        esm_mean_correlations.append(np.nan)
        msat_mean_correlations.append(np.nan)
        continue

    data = np.load(npz_file)["arr_0"]  # Extract array
    num_layers, num_heads, num_species, _ = data.shape

    # === Load and Convert P-Distance Matrix ===
    p_distance_file = f"{base_pdist_folder}/{protein}_34sppAligned_pdist.csv"
    if not os.path.exists(p_distance_file):
        print(f"Skipping {protein}: P-Distance file not found.")
        esm_mean_correlations.append(np.nan)
        msat_mean_correlations.append(np.nan)
        continue

    p_distance_matrix = load_lower_triangular_with_labels(p_distance_file)
    p_distance_flat = p_distance_matrix[np.triu_indices_from(p_distance_matrix, k=1)]

    # === Compute Mean Correlation from ESM ===
    esm_head_correlations = []  # Store correlations for all heads
    for layer in range(num_layers):
        for head in range(num_heads):
            attention_matrix = data[layer, head]
            attention_flat = attention_matrix[np.triu_indices_from(attention_matrix, k=1)]

            if np.all(attention_flat == attention_flat[0]):  # Skip constant values
                continue

            pearson_corr, _ = stats.pearsonr(p_distance_flat, attention_flat)
            esm_head_correlations.append(pearson_corr)

    # Compute mean correlation for ESM
    esm_mean_correlation = np.nanmean(esm_head_correlations) if esm_head_correlations else np.nan
    esm_mean_correlations.append(esm_mean_correlation)

    # === Load MSAT Correlation File ===
    msat_correlation_file = f"{base_msat_folder}/{protein}/{protein}_layerhead_correlation.csv"
    if not os.path.exists(msat_correlation_file):
        print(f"Skipping {protein}: MSAT correlation file not found.")
        msat_mean_correlations.append(np.nan)
        continue

    msat_df = pd.read_csv(msat_correlation_file)
    if "R-Squared" in msat_df.columns:
        msat_mean_correlation = np.sqrt(msat_df["R-Squared"].mean())  # Convert R² to Pearson r
        msat_mean_correlations.append(msat_mean_correlation)
    else:
        print(f"Skipping {protein}: Invalid MSAT correlation file.")
        msat_mean_correlations.append(np.nan)

# === STEP 4: Generate Bar Graph ===
plt.figure(figsize=(10, 6))
x = np.arange(len(protein_list))  # X-axis positions
width = 0.4  # Width of the bars

plt.bar(x - width/2, esm_mean_correlations, width=width, label="ESM-2", color="dodgerblue")
plt.bar(x + width/2, msat_mean_correlations, width=width, label="MSA Transformer", color="orange")

#plt.xlabel("Protein", size = 20)
plt.ylabel("Mean Pearson Correlation", size = 18)
plt.title("Highest Correlation Attention Head vs. P-Distance", size = 20)
plt.xticks(ticks=x, labels=protein_list, rotation=60, size = 14)
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)  # Baseline at 0
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show Plot
plt.show()
