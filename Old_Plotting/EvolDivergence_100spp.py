import numpy as np
import pandas as pd
from Bio import Phylo, SeqIO
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# === USER INPUT ===
protein_name = "CFTR"
distance_npz_file = "/home/rohan/ESM/Results/PSA/Attention_Matrices/CFTR_100_unaligned/CFTR_100_unaligned_distance_matrices.npz"
fasta_file = "/home/rohan/ESM/Data/Alignments/CFTR/CFTR_100_aligned.fas"
tree_file = "/home/rohan/ESM/Data/Trees/UCSC100_Linnaean.nwk"
reference_tip = "Mus_musculus"
binomial_file = "/home/rohan/ESM/Data/UCSC100_Linnaean.txt" # New!


# === STEP 1: Load distance matrix ===
distance_data = np.load(distance_npz_file)
distance_matrix = distance_data["arr_0"]
distance_matrix = np.mean(distance_matrix, axis=(0, 1))  # if it's multi-head

# === STEP 2: Load binomial species names from text file ===
with open(binomial_file, "r") as f:
    binomial_labels = [line.strip() for line in f]
    
print(len(binomial_labels))

# Sanity check: ensure alignment between matrix and label count
assert len(binomial_labels) == distance_matrix.shape[0], \
    "Number of labels must match matrix dimensions."

# === STEP 3: Build labeled DataFrame ===
distance_df = pd.DataFrame(distance_matrix, index=binomial_labels, columns=binomial_labels)

# === STEP 4: Load Newick tree ===
tree = Phylo.read(tree_file, "newick")

# === STEP 5: Calculate divergence times from reference tip ===
def get_divergence_times(tree, reference_tip):
    divergence_times = {}
    ref_depth = tree.distance(tree.root, reference_tip)
    for clade in tree.get_terminals():
        if clade.name == reference_tip:
            continue
        mrca = tree.common_ancestor(reference_tip, clade.name)
        mrca_depth = tree.distance(tree.root, mrca)
        divergence_times[clade.name] = ref_depth - mrca_depth
    return divergence_times

divergence_times = get_divergence_times(tree, reference_tip)
divergence_df = pd.DataFrame(list(divergence_times.items()), columns=["Tip", "Divergence_Time"])

# === STEP 6: Extract distances from reference to all others ===
distance_long = distance_df.stack().reset_index()
distance_long.columns = ["Tip1", "Tip2", "Distance"]
ref_distances = distance_long[distance_long["Tip1"] == reference_tip]

# === STEP 7: Merge and plot ===
merged_df = ref_distances.merge(divergence_df, left_on="Tip2", right_on="Tip")

plt.figure(figsize=(10, 8))
plt.scatter(merged_df["Divergence_Time"], merged_df["Distance"], alpha=0.5, s=150, color="darkgreen")
plt.xlabel(f"Divergence from {reference_tip} in Ma", fontsize=20)
plt.ylabel("Attention distance", fontsize=20)
plt.ylim(0,0.00010)
plt.title(f"{protein_name} attention vs. divergence from {reference_tip}", fontsize=22)

# Optional: annotate key species
annotate_list = ["Pan_troglodytes", "Ornithorhynchus_anatinus", "Monodelphis_domestica", "Sarcophilus_harrisii", "Gallus_gallus", "Takifugu_rubripes", "Petromyzon_marinus"]
for _, row in merged_df.iterrows():
    if row["Tip2"] in annotate_list:
        plt.annotate(row["Tip2"], (row["Divergence_Time"], row["Distance"]),
                     textcoords="offset points", xytext=(0, 10), ha='center',
                     fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))

# Annotate every point with Distance > 0.00020
"""
for _, row in merged_df.iterrows():
    if row["Distance"] > 0.00020:
        plt.annotate(row["Tip2"], (row["Divergence_Time"], row["Distance"]),
                     textcoords="offset points", xytext=(5, 5), ha='left',
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.5))
"""

# === STEP 8: Correlation ===
pearson_r, pearson_p = pearsonr(merged_df["Divergence_Time"], merged_df["Distance"])
spearman_r, spearman_p = spearmanr(merged_df["Divergence_Time"], merged_df["Distance"])

plt.text(0.05, 0.95, f"Pearson R = {pearson_r:.3f}\nP = {pearson_p:.3e}",
         transform=plt.gca().transAxes, fontsize=14,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
         verticalalignment='top')

plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
