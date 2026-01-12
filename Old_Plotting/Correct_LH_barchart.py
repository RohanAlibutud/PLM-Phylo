"""
Fig. B.1.2.3.b | Barchart of correct heads
-------------------------------------------------------------------------------
Percentage of attention heads that were able to infer a quartet tree with an 
accurate topology. Accuracy was assessed by comparing the quartet tree against 
the OrthoMaM supermatrix tree derived from >15K genes, with the “correct” tree 
having a Robinson-Foulds distance of 0 relative to the supermatrix tree.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from RF_scorer import rf_distance

# all proteins
"""
OMAM20_proteins = [
    "AIMP2", "APOBEC2", "AR", "CASQ1", 
    "COCH", "DYRK1B", "GPX3", "HAUS1", 
    "IL6", "INO80B", "KRT23", "NEUROG1",
    "OPN1SW", "PPOX", "RHO", "SLC39A12", 
    "TF", "TPPP2", "UBA2", "UPRT"
]
"""
# all identicals removed
OMAM20_proteins = ["AIMP2",  "AR", "CASQ1", "DYRK1B", "IL6", "INO80B", "KRT23", "NEUROG1", "RHO", "SLC39A12",  "TF",  "UPRT"
]

# all human/chimp identicals removed
#OMAM20_proteins = ["AIMP2", "APOBEC2", "AR", "CASQ1", "COCH",  "GPX3", "INO80B", "KRT23", "NEUROG1", "RHO", "SLC39A12", "TF",  "UPRT"]


# Protein set and supertree path
#protein_set = "4mammals"
protein_set = "4apes"
supertree = "/home/rohan/ESM/Data/Trees/OMAM_supertree.nwk"

# ESM Paths and Parameters
esm_results_folder = f"/home/rohan/ESM/Results/OMAM20/OMAM20_{protein_set}/Trees"
esm_layer_count = 33
esm_head_count = 20

def esm_tree_path(protein, layer, head):
    return f"{esm_results_folder}/{protein}/{protein}/layer{layer}/{protein}_layer{layer}_head{head}_tree.nwk"

# MSAT Paths and Parameters
msat_base_folder = f"/home/rohan/ESM/Results/OMAM20/MSAT/OMAM20_{protein_set}Aligned/Attention_Matrices/"
msat_layer_count = 12
msat_head_count = 12

def msat_tree_path(protein, layer, head):
    return f"{msat_base_folder}{protein}/Trees/aDist_Trees/{protein}_layer{layer}_head{head}_aDist_tree.nwk"

# Function to compute fraction of correct layerheads
def compute_correct_fraction(protein, layer_count, head_count, tree_path_func):
    total_heads = layer_count * head_count
    correct_heads = 0
    
    for layer in tqdm(range(1, layer_count + 1), desc=f"Processing {protein}"):
        for head in range(1, head_count + 1):
            attention_tree = tree_path_func(protein, layer, head)
            if os.path.exists(attention_tree):
                rf = rf_distance(supertree, attention_tree)
                if rf == 0:
                    correct_heads += 1
            else:
                print(f"ERROR: PATH {attention_tree} DOES NOT EXIST")
    
    percentage_correct = correct_heads / total_heads if total_heads > 0 else 0
    print(f"{protein}: {percentage_correct:.2%} ({correct_heads}/{total_heads})")
    return correct_heads, total_heads

# Compute per-protein results
esm_total_correct = 0
esm_total_heads = 0
msat_total_correct = 0
msat_total_heads = 0

esm_correct_fractions = {}
msat_correct_fractions = {}

for protein in OMAM20_proteins:
    esm_correct, esm_heads = compute_correct_fraction(protein, esm_layer_count, esm_head_count, esm_tree_path)
    msat_correct, msat_heads = compute_correct_fraction(protein, msat_layer_count, msat_head_count, msat_tree_path)
    
    esm_total_correct += esm_correct
    esm_total_heads += esm_heads
    msat_total_correct += msat_correct
    msat_total_heads += msat_heads
    
    esm_correct_fractions[protein] = esm_correct / esm_heads if esm_heads > 0 else 0
    msat_correct_fractions[protein] = msat_correct / msat_heads if msat_heads > 0 else 0

# Compute overall correct percentage
esm_overall_percentage = (esm_total_correct / esm_total_heads) * 100 if esm_total_heads > 0 else 0
msat_overall_percentage = (msat_total_correct / msat_total_heads) * 100 if msat_total_heads > 0 else 0

# Print final results
print(f"\nTotal ESM Correct Layerheads: {esm_total_correct}/{esm_total_heads} ({esm_overall_percentage:.2f}%)")
print(f"Total MSAT Correct Layerheads: {msat_total_correct}/{msat_total_heads} ({msat_overall_percentage:.2f}%)\n")

# Plot bar chart
plt.figure(figsize=(10, 6))
x = np.arange(len(OMAM20_proteins))
width = 0.35

plt.bar(x - width/2, [esm_correct_fractions[p] for p in OMAM20_proteins], width, label='ESM', color="dodgerblue")
plt.bar(x + width/2, [msat_correct_fractions[p] for p in OMAM20_proteins], width, label='MSAT', color="orange")

plt.ylabel("Percentage of correct heads", fontsize=18)
plt.title(f"Accuracy of {protein_set} quartet tree inference", fontsize=20)
plt.xticks(ticks=x, labels=OMAM20_proteins, rotation=60, fontsize=14)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
