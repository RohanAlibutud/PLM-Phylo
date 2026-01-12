import numpy as np
import pandas as pd
from Bio import Phylo, SeqIO
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Step 1: Load and average all .npz distance matrices across the folders
base_path = "/home/rohan/ESM/Results/OMAM20/OMAM20_34spp/Attention_Matrices"
folders = [
    "AIMP2", "APOBEC2", "AR", "CASQ1", 
    "COCH", "DYRK1B", "GPX3", "HAUS1", 
    "IL6", "INO80B", "KRT23", "NEUROG1",
    "OPN1SW", "PPOX", "RHO", "SLC39A12", 
    "TF", "TPPP2", "UBA2", "UPRT"
]

all_matrices = []

for folder in folders:
    npz_file = os.path.join(base_path, folder, f"{folder}_distance_matrices.npz")
    distance_data = np.load(npz_file)
    distance_matrix = distance_data["arr_0"]
    all_matrices.append(distance_matrix)

# Step 2: Compute the average matrix across all loaded matrices
combined_average_matrix = np.mean(np.stack(all_matrices), axis=(0, 1, 2))
print(combined_average_matrix.shape)

# Step 3: Parse the FASTA file to extract labels
fasta_file = "/home/rohan/ESM/Data/Alignments/OMAM20/OMAM20_34spp/RHO_34spp.fas"
labels = [record.id for record in SeqIO.parse(fasta_file, "fasta")]

# Step 4: Convert the combined average matrix to a DataFrame with labels
distance_df = pd.DataFrame(combined_average_matrix, index=labels, columns=labels)

# Step 5: Parse the Newick tree
tree = Phylo.read("/home/rohan/ESM/Data/Trees/OMAM20_spp_list.nwk", "newick")

# ✅ Function: Calculate divergence times as depth difference to MRCA
def get_divergence_times(tree, reference_tip="Homo_sapiens"):
    divergence_times = {}
    ref_depth = tree.distance(tree.root, reference_tip)
    for clade in tree.get_terminals():
        if clade.name == reference_tip:
            continue
        mrca = tree.common_ancestor(reference_tip, clade.name)
        mrca_depth = tree.distance(tree.root, mrca)
        divergence_times[clade.name] = ref_depth - mrca_depth
    return divergence_times

# Step 6: Get divergence times
reference_tip = "Homo_sapiens"
divergence_times = get_divergence_times(tree, reference_tip)
divergence_df = pd.DataFrame(list(divergence_times.items()), columns=["Tip", "Divergence_Time"])

# Step 7: Flatten the distance matrix into long format
distance_long = distance_df.stack().reset_index()
distance_long.columns = ["Tip1", "Tip2", "Distance"]

# Step 8: Filter for distances involving Homo_sapiens
ref_distances = distance_long[distance_long["Tip1"] == reference_tip]

# Step 9: Merge with divergence times
merged_df = ref_distances.merge(divergence_df, left_on="Tip2", right_on="Tip")

# Step 10: Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(merged_df["Divergence_Time"], merged_df["Distance"], alpha=0.5, s=150, marker="o", color="darkgreen")
plt.xlabel(f"Divergence from {reference_tip} in Ma", fontsize=20)
plt.ylabel("mean attention distance", fontsize=20)
plt.title(f"ESM-2 attention distance vs. divergence from {reference_tip.replace('_', ' ')}", fontsize=22)

# Step 11: Annotate selected species
specific_species = [
    "Pan_troglodytes", "Gorilla_gorilla_gorilla", "Macaca_fascicularis", 
    "Tupaia_chinensis", "Mus_musculus", "Bos_taurus", "Monodelphis_domestica"
]

for i, row in merged_df.iterrows():
    if row["Tip2"] in specific_species:
        plt.annotate(
            row["Tip2"],
            xy=(row["Divergence_Time"], row["Distance"]),
            xytext=(0, 10),
            textcoords='offset points',
            fontsize=16,
            ha='center',
            color='black',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.7)
        )

# Step 12: Correlation statistics
pearson_r, pearson_p = pearsonr(merged_df["Divergence_Time"], merged_df["Distance"])
spearman_r, spearman_p = spearmanr(merged_df["Divergence_Time"], merged_df["Distance"])

print(f"Pearson's R: {pearson_r:.4f} (p = {pearson_p:.4e})")
print(f"Spearman's ρ: {spearman_r:.4f} (p = {spearman_p:.4e})")

# Step 13: Annotate correlation on the plot
plt.text(
    0.05, 0.95,
    f"Pearson's R: {pearson_r:.3f}\np-value: {pearson_p:.5f}",
    transform=plt.gca().transAxes,
    fontsize=14,
    verticalalignment='top',
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8)
)

plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()
