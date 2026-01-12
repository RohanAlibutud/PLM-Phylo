import numpy as np
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
base_esm_folder = "/home/rohan/ESM/Results/OMAM20/Attention_Matrices"
base_msat_folder = "/home/rohan/ESM/Results/OMAM20/MSAT/OMAM20_4mammalsAligned/Attention_Matrices"

# === Loop over each protein ===
for protein in tqdm(protein_list):

    # === Load ESM Attention Matrices ===
    esm_layer_stdevs = []
    esm_folder = os.path.join(base_esm_folder, f"{protein}_4mammals_Matrices")

    if not os.path.exists(esm_folder):
        print(f"Skipping {protein}: ESM folder not found.")
        continue

    layer_files = sorted(
        [f for f in os.listdir(esm_folder) if f.startswith("layer") and f.endswith(".npz")],
        key=lambda x: int(x.replace("layer", "").replace(".npz", ""))
    )

    if not layer_files:
        print(f"Skipping {protein}: No ESM layer files found.")
        continue

    for layer_file in layer_files:
        layer_path = os.path.join(esm_folder, layer_file)
        data = np.load(layer_path)

        head_vars = []
        for head_key in sorted(data.files):  # e.g., head0, head1, ..., head19
            mat = data[head_key]
            if mat.shape[0] != mat.shape[1]:
                print(f"Warning: Non-square matrix in {layer_file} {head_key}")
                continue

            # Extract upper triangle without the diagonal
            upper_vals = mat[np.triu_indices_from(mat, k=1)]
            variance = np.var(upper_vals)
            head_vars.append(variance)

        if head_vars:
            esm_layer_stdevs.append(np.mean(head_vars))
        else:
            esm_layer_stdevs.append(np.nan)

    esm_layers = len(esm_layer_stdevs)

    # === Compute MSAT variance per layer (only 12 layers expected) ===
    msat_folder = f"{base_msat_folder}/{protein}/{protein}_Matrices/Meanmatrices"
    msat_layers = 12
    msat_heads = 12
    msat_layer_variances = {layer: [] for layer in range(msat_layers)}

    for layer in range(msat_layers):
        for head in range(msat_heads):
            matrix_path = f"{msat_folder}/{protein}_layer{layer+1}_head{head+1}_meanmatrix.csv"
            if os.path.exists(matrix_path):
                matrix = np.loadtxt(matrix_path, delimiter=',')
                variance = np.var(matrix)
                msat_layer_variances[layer].append(variance)
            else:
                print(f"Warning: Missing file {matrix_path}")
                msat_layer_variances[layer].append(np.nan)

    msat_avg_variances = [np.nanmean(msat_layer_variances[layer]) for layer in range(msat_layers)]

    # === Plot Side-by-Side Barcharts ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ESM plot (left)
    axes[0].bar(range(1, esm_layers + 1), esm_layer_stdevs, color='dodgerblue')
    axes[0].set_title(f"ESM Variance per Layer\n{protein}")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean Variance Across Heads")
    axes[0].set_xticks(range(1, esm_layers + 1, 5))
    axes[0].set_xlim(1, esm_layers + 1)

    # MSAT plot (right)
    axes[1].bar(range(1, msat_layers + 1), msat_avg_variances, color='orange')
    axes[1].set_title(f"MSAT Variance per Layer\n{protein}")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean Variance")
    axes[1].set_xticks(range(1, msat_layers + 1))
    axes[1].set_xlim(1, msat_layers + 1)

    plt.suptitle(f"Layer-wise Attention Variance: {protein}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
