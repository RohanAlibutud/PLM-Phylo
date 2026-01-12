import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import statistics

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from RF_scorer import rf_distance

# Protein list
OMAM20_proteins = [
    "AIMP2", "APOBEC2", "AR", "CASQ1", "COCH", "DYRK1B", "GPX3", "HAUS1",
    "IL6", "INO80B", "KRT23", "NEUROG1", "OPN1SW", "PPOX", "RHO", "SLC39A12",
    "TF", "TPPP2", "UBA2", "UPRT"
]

# Tree and config
protein_set = "34spp"
supertree = "/home/rohan/ESM/Data/Trees/OMAM_supertree.nwk"

# ESM config
esm_results_folder = f"/home/rohan/ESM/Results/OMAM20/OMAM20_{protein_set}/Trees"
esm_layer_count = 33
esm_head_count = 20
def esm_tree_path(protein, layer, head):
    return f"{esm_results_folder}/{protein}/{protein}/layer{layer}/{protein}_layer{layer}_head{head}_tree.nwk"

# MSAT config
msat_base_folder = f"/home/rohan/ESM/Results/OMAM20/MSAT/OMAM20_{protein_set}Aligned/Attention_Matrices/"
msat_layer_count = 12
msat_head_count = 12
def msat_tree_path(protein, layer, head):
    return f"{msat_base_folder}{protein}/Trees/aDist_Trees/{protein}_layer{layer}_head{head}_aDist_tree.nwk"

# PC calculator
def calc_PC(RF, num_species=34):
    return round((1 - (RF / 2) / (num_species - 3)) * 100, 1)

# Table containers
esm_summary = []
msat_summary = []

# Gather stats for each protein
for protein in OMAM20_proteins:
    esm_pc_vals = []
    msat_pc_vals = []

    # ESM
    for layer in tqdm(range(1, esm_layer_count + 1), desc=f"ESM - {protein}"):
        for head in range(1, esm_head_count + 1):
            path = esm_tree_path(protein, layer, head)
            if os.path.exists(path):
                rf = rf_distance(supertree, path)
                pc = 100.0 if rf == 0 else calc_PC(rf)
                esm_pc_vals.append(pc)

    if esm_pc_vals:
        esm_summary.append({
            "Protein": protein,
            "Min_PC": min(esm_pc_vals),
            "Max_PC": max(esm_pc_vals),
            "Mean_PC": round(np.mean(esm_pc_vals), 1),
            "Median_PC": round(np.median(esm_pc_vals), 1),
            "Count": len(esm_pc_vals)
        })

    # MSAT
    for layer in tqdm(range(1, msat_layer_count + 1), desc=f"MSAT - {protein}"):
        for head in range(1, msat_head_count + 1):
            path = msat_tree_path(protein, layer, head)
            if os.path.exists(path):
                rf = rf_distance(supertree, path)
                pc = 100.0 if rf == 0 else calc_PC(rf)
                msat_pc_vals.append(pc)

    if msat_pc_vals:
        msat_summary.append({
            "Protein": protein,
            "Min_PC": min(msat_pc_vals),
            "Max_PC": max(msat_pc_vals),
            "Mean_PC": round(np.mean(msat_pc_vals), 1),
            "Median_PC": round(np.median(msat_pc_vals), 1),
            "Count": len(msat_pc_vals)
        })

# Convert to DataFrames and print
esm_df = pd.DataFrame(esm_summary)
msat_df = pd.DataFrame(msat_summary)

print("\n=== ESM Per-Protein Summary ===")
print(esm_df.to_string(index=False))

print("\n=== MSAT Per-Protein Summary ===")
print(msat_df.to_string(index=False))

import matplotlib.pyplot as plt

# Collect data for plotting
esm_box_data = []
msat_box_data = []
labels = []

for protein in tqdm(OMAM20_proteins):
    esm_vals = []
    msat_vals = []

    # ESM
    for layer in range(1, esm_layer_count + 1):
        for head in range(1, esm_head_count + 1):
            path = esm_tree_path(protein, layer, head)
            if os.path.exists(path):
                rf = rf_distance(supertree, path)
                pc = 100.0 if rf == 0 else calc_PC(rf)
                esm_vals.append(pc)

    # MSAT
    for layer in range(1, msat_layer_count + 1):
        for head in range(1, msat_head_count + 1):
            path = msat_tree_path(protein, layer, head)
            if os.path.exists(path):
                rf = rf_distance(supertree, path)
                pc = 100.0 if rf == 0 else calc_PC(rf)
                msat_vals.append(pc)

    esm_box_data.append(esm_vals)
    msat_box_data.append(msat_vals)
    labels.append(protein)

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
positions_esm = np.arange(len(labels)) * 2
positions_msat = positions_esm + 0.6

# ESM boxes
b1 = ax.boxplot(esm_box_data, positions=positions_esm, widths=0.5, patch_artist=True,
                boxprops=dict(facecolor="dodgerblue", color="black"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                flierprops=dict(markerfacecolor="dodgerblue", markeredgecolor="black", markersize=3))

# MSAT boxes
b2 = ax.boxplot(msat_box_data, positions=positions_msat, widths=0.5, patch_artist=True,
                boxprops=dict(facecolor="orange", color="black"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                flierprops=dict(markerfacecolor="orange", markeredgecolor="black", markersize=3))

# X-axis setup
ax.set_xticks(positions_esm + 0.3)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=16)
ax.set_ylabel("Percent Correct (%)", fontsize=16)
ax.set_title("Percentage of correctly inferred clades (PC)", fontsize=20)
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.legend([b1["boxes"][0], b2["boxes"][0]], ["PLM-ss", "PLM-msa"], loc="upper left", fontsize = 16)

plt.tight_layout()
plt.show()
