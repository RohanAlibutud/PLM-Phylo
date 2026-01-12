import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr

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

esm_layer_count = 36 #33
msat_layer_count = 12
esm_head_count = 60 #20
msat_head_count = 12

# === Storage: list of per-protein variance arrays ===
esm_all_proteins = []  # each item: [var_l1, var_l2, ..., var_l33]
msat_all_proteins = [] # each item: [var_l1, ..., var_l12]

for protein in protein_list:
    print(f"\nProcessing {protein}...")

    # === ESM ===
    esm_folder = os.path.join(base_esm_folder, f"{protein}_4mammals_Matrices")
    esm_variances = []

    if os.path.exists(esm_folder):
        layer_files = sorted(
            [f for f in os.listdir(esm_folder) if f.startswith("layer") and f.endswith(".npz")],
            key=lambda x: int(x.replace("layer", "").replace(".npz", ""))
        )

        for layer_file in layer_files:
            data = np.load(os.path.join(esm_folder, layer_file))
            head_vars = []
            for head_key in sorted(data.files):
                mat = data[head_key]
                vals = mat[np.triu_indices_from(mat, k=1)]
                head_vars.append(np.var(vals))
            esm_variances.append(np.mean(head_vars))

        # Pad if fewer than 33 layers (e.g., model failed to save all)
        if len(esm_variances) < esm_layer_count:
            esm_variances += [np.nan] * (esm_layer_count - len(esm_variances))

    else:
        esm_variances = [np.nan] * esm_layer_count
        print(f"  ESM missing for {protein}")

    esm_all_proteins.append(esm_variances)

    # === MSAT ===
    msat_folder = f"{base_msat_folder}/{protein}/{protein}_Matrices/Meanmatrices"
    msat_variances = []

    for layer in range(msat_layer_count):
        head_vars = []
        for head in range(msat_head_count):
            matrix_path = f"{msat_folder}/{protein}_layer{layer+1}_head{head+1}_meanmatrix.csv"
            if os.path.exists(matrix_path):
                matrix = np.loadtxt(matrix_path, delimiter=',')
                head_vars.append(np.var(matrix))
        msat_variances.append(np.mean(head_vars) if head_vars else np.nan)

    msat_all_proteins.append(msat_variances)

# === Convert to NumPy for easier averaging ===
esm_all_proteins = np.array(esm_all_proteins)  # shape: (20, 33)
msat_all_proteins = np.array(msat_all_proteins)  # shape: (20, 12)

# === Compute mean per layer ===
esm_mean_variance = np.nanmean(esm_all_proteins, axis=0)
msat_mean_variance = np.nanmean(msat_all_proteins, axis=0)

# === Plotting ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ESM Plot
axes[0].bar(range(1, esm_layer_count + 1), esm_mean_variance, color='dodgerblue')
axes[0].set_title("Average ESM Attention Variance Across Proteins")
axes[0].set_xlabel("Layer")
axes[0].set_ylabel("Mean Variance")
axes[0].set_xticks(range(1, esm_layer_count + 1, 5))
axes[0].set_xlim(1, esm_layer_count + 1)

# MSAT Plot
axes[1].bar(range(1, msat_layer_count + 1), msat_mean_variance, color='orange')
axes[1].set_title("Average MSAT Attention Variance Across Proteins")
axes[1].set_xlabel("Layer")
axes[1].set_ylabel("Mean Variance")
axes[1].set_xticks(range(1, msat_layer_count + 1))
axes[1].set_xlim(1, msat_layer_count + 1)

plt.suptitle("Layer-wise Mean Attention Variance (Averaged Across Proteins)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# report correlation
esm_layers = np.arange(1, len(esm_mean_variance) + 1)
esm_r, esm_p = pearsonr(esm_layers, esm_mean_variance)
print(f"ESM att variance over ALL layers: R = {esm_r:.3f}, p = {esm_p:.9f}")

esm_r, esm_p = pearsonr(esm_layers[6:], esm_mean_variance[6:])
print(f"ESM att variance over layers > 6: R = {esm_r:.3f}, p = {esm_p:.9f}")

msat_layers = np.arange(1, len(msat_mean_variance) + 1)
msat_r, msat_p = pearsonr(msat_layers, msat_mean_variance)
print(f"MSAT att variance over layers: R = {msat_r:.3f}, p = {msat_p:.9f}")
