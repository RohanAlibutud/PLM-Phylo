import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# === Protein list ===
protein_list = [
    "AIMP2", "APOBEC2", "AR", "CASQ1", 
    "COCH", "DYRK1B", "GPX3", "HAUS1", 
    "IL6", "INO80B", "KRT23", "NEUROG1",
    "OPN1SW", "PPOX", "RHO", "SLC39A12", 
    "TF", "TPPP2", "UBA2", "UPRT"
]

# === Base folders ===
base_esm_folder = "/home/rohan/ESM/Results/OMAM20/OMAM20_34spp/Attention_Matrices"
base_pdist_folder = "/home/rohan/ESM/Results/MEGA_Pdist_Matrices/34sppAligned"
plots_folder = "/home/rohan/ESM/Results/OMAM20/Plots"

# === Helper function ===
def load_lower_triangular_with_labels(file_path):
    df = pd.read_csv(file_path)
    species = df.iloc[:, 0].values
    df_values = df.iloc[:, 1:].values.astype(float)
    size = len(species)
    full_matrix = np.zeros((size, size))
    for i in range(size):
        full_matrix[i, :i+1] = df_values[i, :i+1]
        full_matrix[:i+1, i] = df_values[i, :i+1]
    return full_matrix

# === Store all correlation matrices ===
all_correlation_matrices = []

for idx, protein in enumerate(protein_list):
    print(f"\nProcessing {protein}...")

    npz_file = os.path.join(base_esm_folder, f"{protein}/{protein}_distance_matrices.npz")
    pdist_file = os.path.join(base_pdist_folder, f"{protein}_34sppAligned_pdist.csv")

    try:
        data = np.load(npz_file)["arr_0"]
    except FileNotFoundError:
        print(f"Warning: NPZ file not found for {protein}, skipping.")
        continue

    assert data.shape == (33, 20, 34, 34), f"Unexpected shape {data.shape} for {protein}"

    try:
        p_distance_matrix = load_lower_triangular_with_labels(pdist_file)
    except FileNotFoundError:
        print(f"Warning: P-Distance file not found for {protein}, skipping.")
        continue

    assert p_distance_matrix.shape == (34, 34), f"Unexpected shape: {p_distance_matrix.shape}"

    p_flat = p_distance_matrix[np.triu_indices_from(p_distance_matrix, k=1)]
    corr_matrix = np.full((33, 20), np.nan)

    for layer in tqdm(range(33), desc=f"Layers for {protein}"):
        for head in range(20):
            att_matrix = data[layer, head]
            att_flat = att_matrix[np.triu_indices_from(att_matrix, k=1)]

            if np.all(att_flat == att_flat[0]):
                continue

            corr, _ = stats.pearsonr(p_flat, att_flat)
            corr_matrix[layer, head] = corr

    all_correlation_matrices.append(corr_matrix)

    # === Plot protein-specific heatmap ===
    plt.figure(figsize=(7, 4))
    im = plt.imshow(corr_matrix.T, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, label='Pearson Correlation')
    plt.xlabel('Layer')
    plt.ylabel('Head')
    plt.title(f'Correlation Heatmap: {protein}')

    # Integer ticks (1-indexed)
    plt.xticks(ticks=np.arange(0, 33, 3), labels=np.arange(1, 34, 3))
    plt.yticks(ticks=np.arange(0, 20, 2), labels=np.arange(1, 21, 2))
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/{protein}_correlation_heatmap.png")
    plt.show()
    plt.close()

# === Compute and plot average heatmap ===

# MEAN
average_matrix = np.nanmean(np.array(all_correlation_matrices), axis=0)
plt.figure(figsize=(7, 4))
im = plt.imshow(average_matrix.T, aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
plt.colorbar(im, label='Mean Pearson Correlation')
plt.xlabel('Layer')
plt.ylabel('Head')
plt.title('Mean Correlation Heatmap (All Proteins)')
plt.xticks(ticks=np.arange(0, 33, 3), labels=np.arange(1, 34, 3))
plt.yticks(ticks=np.arange(0, 20, 2), labels=np.arange(1, 21, 2))
plt.tight_layout()
plt.savefig(f"{plots_folder}/ESM_mean_correlation_heatmap.png")
plt.show()

# MEDIAN
average_matrix = np.nanmedian(np.array(all_correlation_matrices), axis=0)
plt.figure(figsize=(7, 4))
im = plt.imshow(average_matrix.T, aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
plt.colorbar(im, label='median Pearson Correlation')
plt.xlabel('Layer', fontsize=14)
plt.ylabel('Head', fontsize=14)
plt.title('Median Correlation Heatmap (All Proteins)', fontsize=16)
plt.xticks(ticks=np.arange(0, 33, 3), labels=np.arange(1, 34, 3))
plt.yticks(ticks=np.arange(0, 20, 2), labels=np.arange(1, 21, 2))
plt.tight_layout()
plt.savefig(f"{plots_folder}/ESM_median_correlation_heatmap.png")
plt.show()
