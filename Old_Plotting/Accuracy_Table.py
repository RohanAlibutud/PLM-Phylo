"""
Table 1: Accuracy of inferred trees by ESM and MSAT 
-------------------------------------------------------------------------------
Accuracy reported in terms of max, mean, early, and late. IQTREE ML by best fit
model also included for comparison.
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from RF_scorer import rf_distance


# === Define the list of proteins ===
OMAM20_proteins = [
    "AIMP2", "APOBEC2", "AR", "CASQ1", 
    "COCH", "DYRK1B", "GPX3", "HAUS1", 
    "IL6", "INO80B", "KRT23", "NEUROG1",
    "OPN1SW", "PPOX", "RHO", "SLC39A12", 
    "TF", "TPPP2", "UBA2", "UPRT"
]


# === Define base folders ===
protein_set = "4apes"

# Define results directories
supertree = "/home/rohan/ESM/Data/Trees/OMAM_supertree.nwk"
esm_results_folder = f"/home/rohan/ESM/Results/OMAM20/OMAM20_{protein_set}/Trees"
msat_results_folder = f"/home/rohan/ESM/Results/OMAM20/MSAT/OMAM20_{protein_set}Aligned/Attention_Matrices"
iqtree_results_folder = f"/home/rohan/ESM/Results/OMAM20/ML_MFP/OMAM20_{protein_set}Aligned"


layer_count = 33
head_count = 20

# Initialize accuracy table
data = []

for protein in OMAM20_proteins:
    print(f"Processing {protein}...\n")
    # ESM Accuracy Calculation
    esm_layer_accuracies = []
    for layer_index in tqdm(range(1, layer_count + 1), desc = "ESM layers"):
        correct_heads = 0
        total_heads = 0
        for head_index in range(1, head_count + 1):
            esm_tree = f"{esm_results_folder}/{protein}/{protein}/layer{layer_index}/{protein}_layer{layer_index}_head{head_index}_tree.nwk"
            if os.path.exists(esm_tree):
                rf = rf_distance(supertree, esm_tree)
                if rf == 0:
                    correct_heads += 1
                total_heads += 1
            else:
                print("ESM TREE NO EXISTE")
                print(esm_tree)
                break
        if total_heads > 0:
            esm_layer_accuracies.append(correct_heads / total_heads * 100)
    esm_max_acc = max(esm_layer_accuracies, default=0)
    esm_avg_acc = np.mean(esm_layer_accuracies) if esm_layer_accuracies else 0
    esm_early_acc = np.mean(esm_layer_accuracies[:11]) # first third of ESM-2 650M
    esm_late_acc = np.mean(esm_layer_accuracies[22:]) # last third of ESM-2 650M
 
    # MSAT Accuracy Calculation
    msat_layer_accuracies = []
    msat_folder = f"{msat_results_folder}/{protein}/Trees/aDist_Trees"
    for layer_index in tqdm(range(1, 12 + 1), desc = "MSAT layers"):
        correct_heads = 0
        total_heads = 0
        for head_index in range(1, 12 + 1):
            msat_tree = f"{msat_folder}/{protein}_layer{layer_index}_head{head_index}_aDist_tree.nwk"
            if os.path.exists(msat_tree):
                rf = rf_distance(supertree, msat_tree)
                if rf == 0:
                    correct_heads += 1
                total_heads += 1
            else:
                print("MSAT TREE NO EXISTE")
                print(msat_tree)
                break
        if total_heads > 0:
            msat_layer_accuracies.append(correct_heads / total_heads * 100)
    msat_max_acc = max(msat_layer_accuracies, default=0)
    msat_avg_acc = np.mean(msat_layer_accuracies) if msat_layer_accuracies else 0
    msat_early_acc = np.mean(msat_layer_accuracies[:4]) # first third of MSAT
    msat_late_acc = np.mean(msat_layer_accuracies[8:]) # last third of MSAT

    # IQ-TREE Accuracy Calculation
    iqtree_tree = f"{iqtree_results_folder}/{protein}_{protein_set}Aligned.treefile"
    if os.path.exists(iqtree_tree):
        rf = rf_distance(supertree, msat_tree)
        if rf == 0:
            correct_heads += 1
        total_heads += 1
    else:
        print("ML-MFP TREE NO EXISTE")
        print(iqtree_tree)
        break
    iqtree_rf = rf_distance(supertree, iqtree_tree)
    print(f"IQTREE RF FOR {protein}: {iqtree_rf}")
    iqtree_acc = 100 if os.path.exists(iqtree_tree) and rf_distance(supertree, iqtree_tree) == 0 else 0

    # Store results
    data.append([protein, esm_max_acc, esm_avg_acc, esm_early_acc, esm_late_acc, msat_max_acc, msat_avg_acc, msat_early_acc, msat_late_acc, iqtree_acc])

# Convert to DataFrame and display
columns = ["Protein", "ESM Max Acc", "ESM Avg Acc", "ESM Early Acc", "ESM Late Acc", "MSAT Max Acc", "MSAT Avg Acc", "MSAT Early Acc", "MSAT Late Acc", "IQ-TREE Acc"]
df = pd.DataFrame(data, columns=columns)
df = df.round(1)
print(df)

df.to_csv(f"/home/rohan/ESM/Results/OMAM20_{protein_set}_accuracy_table.csv", index=False)
