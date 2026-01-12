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
base_esm_folder = "/home/rohan/ESM/Results/OMAM20/OMAM20_4mammals/Attention_Matrices"

# === Loop over each protein ===
for protein in protein_list:
    print(f"\nProcessing {protein}...")

    # === Load ESM Attention Distance Matrices ===
    npz_file = f"{base_esm_folder}/{protein}/{protein}_distance_matrices.npz"
    if not os.path.exists(npz_file):
        print(f"Skipping {protein}: ESM matrix file not found.")
        continue

    data = np.load(npz_file)["arr_0"]  # shape: (layers, heads, species, species)
    num_layers, num_heads, num_species, _ = data.shape

    # === Store per-layer standard deviations ===
    layer_stdevs = []

    for layer in range(num_layers):
        head_values = []

        for head in range(num_heads):
            # Extract upper triangle values of the attention matrix
            attn = data[layer, head]
            attn_flat = attn[np.triu_indices_from(attn, k=1)]
            head_values.append(attn_flat)

        # Convert to array: (num_heads, num_values_per_head)
        head_values = np.array(head_values)

        # Std dev across heads for each value in the matrix, then mean to summarize
        stdev_across_heads = np.std(head_values, axis=0)
        mean_stdev = np.mean(stdev_across_heads)
        layer_stdevs.append(mean_stdev)

    # === Barplot ===
    plt.figure(figsize=(10, 5))
    plt.bar(range(num_layers), layer_stdevs, color='dodgerblue')
    plt.title(f'Stdev of Attention Across Heads - {protein}')
    plt.xlabel('Layer')
    plt.ylabel('Mean Std. Dev.')
    plt.xticks(range(num_layers))
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()