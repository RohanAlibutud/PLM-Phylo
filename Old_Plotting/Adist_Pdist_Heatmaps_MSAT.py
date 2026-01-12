import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# === Protein List ===
OMAM20_proteins = [
    "AIMP2", "APOBEC2", "AR", "CASQ1", 
    "COCH", "DYRK1B", "GPX3", "HAUS1", 
    "IL6", "INO80B", "KRT23", "NEUROG1",
    "OPN1SW", "PPOX", "RHO", "SLC39A12", 
    "TF", "TPPP2", "UBA2", "UPRT"
]

# === MSAT CSV Folder Path ===
base_msat_folder = "/home/rohan/ESM/Results/OMAM20/MSAT/OMAM20_34sppAligned/Attention_Matrices"
plots_folder = "/home/rohan/ESM/Results/OMAM20/Plots"

# === Storage for All Matrices ===
all_msat_matrices = []

# === Loop Over Proteins ===
for protein in tqdm(OMAM20_proteins, desc="Processing MSAT Heatmaps"):
    file_path = os.path.join(base_msat_folder, protein, f"{protein}_layerhead_correlation.csv")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Warning: File not found for {protein}")
        continue

    df_sorted = df.sort_values(by=["Layer", "Head"])
    
    # Initialize 12x12 matrix (Heads x Layers)
    matrix = np.full((12, 12), np.nan)

    for _, row in df_sorted.iterrows():
        layer = int(row["Layer"]) - 1  # Convert to 0-based index
        head = int(row["Head"]) - 1
        matrix[head, layer] = row["R-Squared"]

    all_msat_matrices.append(matrix)

    # === Plot Heatmap for This Protein ===
    plt.figure(figsize=(8, 6))
    im = plt.imshow(matrix, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, label='R-Squared')

    # One-indexed tick labels, spaced
    plt.xticks(ticks=np.arange(0, 12, 3), labels=np.arange(1, 13, 3))
    plt.yticks(ticks=np.arange(0, 12, 2), labels=np.arange(1, 13, 2))
    
    plt.xlabel("Layer", fontsize=14)
    plt.ylabel("Head", fontsize=14)
    plt.title(f"MSAT/Pdist Heatmap: {protein}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/{protein}_msat_heatmap.png")
    plt.show()
    plt.close()

# === Plot Average MSAT Heatmap ===
# MEAN
avg_matrix = np.nanmean(np.array(all_msat_matrices), axis=0)

plt.figure(figsize=(10, 8))
im = plt.imshow(avg_matrix, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
plt.colorbar(im, label='Mean R-Squared')

plt.xticks(ticks=np.arange(0, 12, 1), labels=np.arange(1, 13, 1))
plt.yticks(ticks=np.arange(0, 12, 1), labels=np.arange(1, 13, 1))
plt.xlabel("Layer", fontsize=14)
plt.ylabel("Head", fontsize=14)
plt.title("Mean MSAT/Pdist Heatmap (All Proteins)", fontsize=16)
plt.tight_layout()
plt.savefig(f"{plots_folder}/MSAT_mean_heatmap.png")
plt.show()

# MEDIAN
avg_matrix = np.nanmedian(np.array(all_msat_matrices), axis=0)

plt.figure(figsize=(5, 4))
im = plt.imshow(avg_matrix, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
plt.colorbar(im, label='median R-Squared')

plt.xticks(ticks=np.arange(0, 12, 1), labels=np.arange(1, 13, 1))
plt.yticks(ticks=np.arange(0, 12, 1), labels=np.arange(1, 13, 1))
plt.xlabel("Layer", fontsize=14)
plt.ylabel("Head", fontsize=14)
plt.title("median MSAT/Pdist Heatmap (All Proteins)", fontsize=16)
plt.tight_layout()
plt.savefig(f"{plots_folder}/MSAT_median_heatmap.png")
plt.show()


