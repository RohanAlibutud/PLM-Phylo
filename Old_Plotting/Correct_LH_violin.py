"""
Fig. B.2.1 | Violin plot of correct heads
-------------------------------------------------------------------------------
a) Percentage of attention heads that produced the correct tree for a quartet 
of distantly related mammals (human, macaque, mouse, opossum). 
b) Percentage of attention heads that produced the correct tree for a quartet 
of more closely related primates (human, chimpanzee, macaque, spider monkey).
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from RF_scorer import rf_distance

OMAM20_proteins = [
    "AIMP2", "APOBEC2", "AR", "CASQ1", 
    "COCH", "DYRK1B", "GPX3", "HAUS1", 
    "IL6", "INO80B", "KRT23", "NEUROG1",
    "OPN1SW", "PPOX", "RHO", "SLC39A12", 
    "TF", "TPPP2", "UBA2", "UPRT"
]

supertree = "/home/rohan/ESM/Data/Trees/OMAM_supertree.nwk"
protein_set  = "mammals"
#protein_set  = "primates"
results_folder = f"/home/rohan/ESM/Results/OMAM20/OMAM20_4{protein_set}/Trees/"
esm_results_folder = "/home/rohan/ESM/Results/OMAM20/OMAM20_4mammals/Trees"
layer_count = 33
head_count = 20

# Dictionary to store correct percentages per layer for each protein
protein_layer_correct = {protein: [] for protein in OMAM20_proteins}

for protein in OMAM20_proteins:
    for layer in tqdm(range(1, layer_count + 1)):
        heads_correct = 0
        heads_incorrect = 0
        
        for head in range(1, head_count + 1):
            attention_tree = f"{esm_results_folder}/{protein}/{protein}/layer{layer}/{protein}_layer{layer}_head{head}_tree.nwk"
            rf = rf_distance(supertree, attention_tree)
            
            if rf == 0:
                heads_correct += 1
            else:
                heads_incorrect += 1

        percent_correct = heads_correct / (heads_correct + heads_incorrect)
        protein_layer_correct[protein].append(percent_correct)

# Convert to a list format for violin plot
data = [protein_layer_correct[protein] for protein in OMAM20_proteins]

# Plot Violin Plot
plt.figure(figsize=(10, 6))
vp = plt.violinplot(data, showmeans=True, showmedians=True, widths=0.5)
for pc in vp['bodies']:
    pc.set_facecolor('dodgerblue')
    pc.set_alpha(0.6)

# Set x-axis labels and title
plt.xticks(ticks=np.arange(1, len(OMAM20_proteins) + 1), labels=OMAM20_proteins, rotation=60, fontsize = 14)
plt.ylabel("Percentage of correct heads", fontsize = 18)
plt.title(f"Correct heads across layers per protein for {protein_set}", fontsize = 20)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()
