import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Extend path for rf_distance
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from RF_scorer import rf_distance

# Define proteins
OMAM20_proteins = [
    "AIMP2", "APOBEC2", "AR", "CASQ1", "COCH", "DYRK1B", "GPX3", "HAUS1",
    "IL6", "INO80B", "KRT23", "NEUROG1", "OPN1SW", "PPOX", "RHO", "SLC39A12",
    "TF", "TPPP2", "UBA2", "UPRT"
]

# Parameters
protein_set = "34spp"
supertree = "/home/rohan/ESM/Data/Trees/OMAM_supertree.nwk"

# Layer & head counts
esm_layer_count, esm_head_count = 33, 20
msat_layer_count, msat_head_count = 12, 12

# Path builders
def esm_tree_path(protein, layer, head):
    return f"/home/rohan/ESM/Results/OMAM20/OMAM20_{protein_set}/Trees/{protein}/{protein}/layer{layer}/{protein}_layer{layer}_head{head}_tree.nwk"

def msat_tree_path(protein, layer, head):
    return f"/home/rohan/ESM/Results/OMAM20/MSAT/OMAM20_{protein_set}Aligned/Attention_Matrices/{protein}/Trees/aDist_Trees/{protein}_layer{layer}_head{head}_aDist_tree.nwk"

def esm_consensus_path(protein):
    return f"/home/rohan/ESM/Results/OMAM20/OMAM20_34spp/Trees/{protein}/Consensus_Trees/{protein}_astral_consensus_tree.nwk"

def msat_consensus_path(protein):
    return f"/home/rohan/ESM/Results/OMAM20/MSAT/OMAM20_{protein_set}Aligned/Attention_Matrices/{protein}/{protein}_astral_output.nwk"

# % Correct calculator
def calc_PC(RF, num_species=34):
    return round((1 - (RF / 2) / (num_species - 3)) * 100, 1)

# Containers
esm_box_data = []
msat_box_data = []
esm_consensus_PC = []
msat_consensus_PC = []

# Collect data
for protein in OMAM20_proteins:
    esm_vals = []
    msat_vals = []

    # ESM layer-head trees
    for layer in tqdm(range(1, esm_layer_count + 1), desc=f"ESM - {protein}"):
        for head in range(1, esm_head_count + 1):
            path = esm_tree_path(protein, layer, head)
            if os.path.exists(path):
                rf = rf_distance(supertree, path)
                pc = 100.0 if rf == 0 else calc_PC(rf)
                esm_vals.append(pc)

    # MSAT layer-head trees
    for layer in tqdm(range(1, msat_layer_count + 1), desc=f"MSAT - {protein}"):
        for head in range(1, msat_head_count + 1):
            path = msat_tree_path(protein, layer, head)
            if os.path.exists(path):
                rf = rf_distance(supertree, path)
                pc = 100.0 if rf == 0 else calc_PC(rf)
                msat_vals.append(pc)

    esm_box_data.append(esm_vals)
    msat_box_data.append(msat_vals)

    # ESM consensus
    consensus_path = esm_consensus_path(protein)
    if os.path.exists(consensus_path):
        rf = rf_distance(supertree, consensus_path)
        pc = 100.0 if rf == 0 else calc_PC(rf)
        esm_consensus_PC.append(pc)
    else:
        print(f"Missing ESM consensus for {protein}")
        esm_consensus_PC.append(None)

    # MSAT consensus
    consensus_path = msat_consensus_path(protein)
    if os.path.exists(consensus_path):
        rf = rf_distance(supertree, consensus_path)
        pc = 100.0 if rf == 0 else calc_PC(rf)
        msat_consensus_PC.append(pc)
    else:
        print(f"Missing MSAT consensus for {protein}")
        msat_consensus_PC.append(None)

# ========== Plotting ==========
positions_esm = np.arange(len(OMAM20_proteins)) * 2
positions_msat = positions_esm + 0.6

fig, ax = plt.subplots(figsize=(14, 6))

# ESM boxplot
b1 = ax.boxplot(esm_box_data, positions=positions_esm, widths=0.5, patch_artist=True,
                boxprops=dict(facecolor="dodgerblue", color="black"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                flierprops=dict(markerfacecolor="dodgerblue", markeredgecolor="black", markersize=3))

# MSAT boxplot
b2 = ax.boxplot(msat_box_data, positions=positions_msat, widths=0.5, patch_artist=True,
                boxprops=dict(facecolor="orange", color="black"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                flierprops=dict(markerfacecolor="orange", markeredgecolor="black", markersize=3))

# Consensus markers
for i, (pc_esm, pc_msat) in enumerate(zip(esm_consensus_PC, msat_consensus_PC)):
    if pc_esm is not None:
        ax.plot(positions_esm[i], pc_esm,
                marker="*", color="dodgerblue", markersize=12,
                markeredgecolor="mediumblue", markeredgewidth=1,
                label="PLM-ss consensus" if i == 0 else "")
    if pc_msat is not None:
        ax.plot(positions_msat[i], pc_msat,
                marker="*", color="orange", markersize=12,
                markeredgecolor="peru", markeredgewidth=1,
                label="PLM-msa consensus" if i == 0 else "")

# Labels and aesthetics
ax.set_xticks(positions_esm + 0.3)
ax.set_xticklabels(OMAM20_proteins, rotation=45, ha="right", fontsize=16)
ax.set_ylabel("Percent Correct (%)", fontsize=16)
ax.set_title("Percentage of correctly inferred clades (PC), PLM-ss vs. PLM-msa", fontsize=20)
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Custom legend
handles = [
    b1["boxes"][0],
    b2["boxes"][0],
    plt.Line2D([0], [0], marker="*", color="dodgerblue", linestyle="", label="PLM-ss consensus"),
    plt.Line2D([0], [0], marker="*", color="orange", linestyle="", label="PLM-msa consensus")
]
ax.legend(handles=handles, loc="upper left")

plt.tight_layout()
plt.show()
