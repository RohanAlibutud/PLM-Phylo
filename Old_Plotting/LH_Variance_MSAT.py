import numpy as np
import pandas as pd
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
base_msat_folder = "/home/rohan/ESM/Results/OMAM20/MSAT/OMAM20_4mammalsAligned/Attention_Matrices"

# === Loop over each protein ===
for protein in protein_list:
    print(f"\nProcessing {protein}...")

    # === Load msat Attention Distance Matrices ===
    meanmatrix_folder = f"{base_msat_folder}/{protein}/{protein}_Matrices/Meanmatrices"
    
    num_layers = 12
    num_heads = 12
    
    # Store per-head variances in a layer-wise dict
    layer_variances = {layer: [] for layer in range(num_layers)}

    for layer in range(num_layers):
        for head in range(num_heads):
            matrix_path = f"{meanmatrix_folder}/{protein}_layer{layer+1}_head{head+1}_meanmatrix.csv"
            if os.path.exists(matrix_path):
                matrix = np.loadtxt(matrix_path, delimiter=',')
                variance = np.var(matrix)
                layer_variances[layer].append(variance)
            else:
                print(f"Warning: Missing file {matrix_path}")
                layer_variances[layer].append(np.nan)  # Optional: mark missing with NaN

    # === Compute average variance per layer across heads ===
    layer_avg_variance = [np.nanmean(layer_variances[layer]) for layer in range(num_layers)]

    # === Barplot ===
    plt.figure(figsize=(10, 5))
    plt.bar(range(num_layers), layer_avg_variance, color='orange')
    plt.title(f'Mean Variance of Attention Across Heads - {protein}')
    plt.xlabel('Layer')
    plt.ylabel('Mean Variance')
    plt.xticks(range(num_layers))
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
