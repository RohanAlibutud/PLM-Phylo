import numpy as np
import matplotlib.pyplot as plt
import os

# === Protein List ===
protein_list = [
    "AIMP2", "APOBEC2", "AR", "CASQ1", 
    "COCH", "DYRK1B", "GPX3", "HAUS1", 
    "IL6", "INO80B", "KRT23", "NEUROG1",
    "OPN1SW", "PPOX", "RHO", "SLC39A12", 
    "TF", "TPPP2", "UBA2", "UPRT"
]

# === Base Paths ===
base_esm_folder = "/home/rohan/ESM/Results/OMAM20/Attention_Matrices/4mammals"
base_msat_folder = "/home/rohan/ESM/Results/OMAM20/MSAT/OMAM20_4mammalsAligned/Attention_Matrices"

esm_layer_count = 36  # total expected layers
esm_head_count = 40
msat_layer_count = 12
msat_head_count = 12

# === Accumulators ===
esm_matrix_all = np.full((esm_layer_count, esm_head_count, len(protein_list)), np.nan)
msat_matrix_all = np.full((msat_layer_count, msat_head_count, len(protein_list)), np.nan)

for p_idx, protein in enumerate(protein_list):
    print(f"\nProcessing {protein}...")

    # --- ESM ---
    esm_folder = os.path.join(base_esm_folder, f"{protein}_4mammals_Matrices")
    if os.path.exists(esm_folder):
        layer_files = sorted(
            [f for f in os.listdir(esm_folder) if f.startswith("layer") and f.endswith(".npz")],
            key=lambda x: int(x.replace("layer", "").replace(".npz", ""))
        )
        for l_idx, layer_file in enumerate(layer_files):
            layer_path = os.path.join(esm_folder, layer_file)
            data = np.load(layer_path)
            for h_idx, head_key in enumerate(sorted(data.files)):
                mat = data[head_key]
                vals = mat[np.triu_indices_from(mat, k=1)]
                esm_matrix_all[l_idx, h_idx, p_idx] = np.std(vals)
    else:
        print(f"  ESM folder not found for {protein}")

    # --- MSAT ---
    msat_folder = f"{base_msat_folder}/{protein}/{protein}_Matrices/Meanmatrices"
    for l_idx in range(msat_layer_count):
        for h_idx in range(msat_head_count):
            matrix_path = f"{msat_folder}/{protein}_layer{l_idx+1}_head{h_idx+1}_meanmatrix.csv"
            if os.path.exists(matrix_path):
                mat = np.loadtxt(matrix_path, delimiter=',')
                msat_matrix_all[l_idx, h_idx, p_idx] = np.std(mat)

# === Compute Mean Std Dev per (layer, head) ===
esm_mean_std = np.nanmean(esm_matrix_all, axis=2)  # shape: (36, 60)
msat_mean_std = np.nanmean(msat_matrix_all, axis=2)  # shape: (12, 12)

# === Plot Heatmaps ===
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# ESM heatmap
im1 = axes[0].imshow(esm_mean_std.T, cmap='plasma', aspect='auto')
axes[0].set_title("ESM Mean Std. Dev. of Attention Values")
axes[0].set_xlabel("Layer (1–36)")
axes[0].set_ylabel("Head (1–60)")
axes[0].set_xticks(np.arange(0, esm_layer_count, 5))
axes[0].set_xticklabels(np.arange(1, esm_layer_count+1, 5))
axes[0].set_yticks(np.arange(0, esm_head_count, 5))
axes[0].set_yticklabels(np.arange(1, esm_head_count+1, 5))
plt.colorbar(im1, ax=axes[0], label='Std. Dev.')

# MSAT heatmap
im2 = axes[1].imshow(msat_mean_std.T, cmap='plasma', aspect='auto')
axes[1].set_title("MSAT Mean Std. Dev. of Attention Values")
axes[1].set_xlabel("Layer (1–12)")
axes[1].set_ylabel("Head (1–12)")
axes[1].set_xticks(np.arange(0, msat_layer_count, 2))
axes[1].set_xticklabels(np.arange(1, msat_layer_count+1, 2))
axes[1].set_yticks(np.arange(0, msat_head_count, 2))
axes[1].set_yticklabels(np.arange(1, msat_head_count+1, 2))
plt.colorbar(im2, ax=axes[1], label='Std. Dev.')

plt.tight_layout()
plt.savefig("attention_std_heatmaps.png")
plt.show()
