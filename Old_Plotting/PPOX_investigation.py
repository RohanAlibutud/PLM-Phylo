import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm

OMAM10_proteins = ["PPOX"]
protein_set = "4mammals"

# === STEP 1: Load Attention Distance Matrices ===
npz_folder = f"/home/rohan/ESM/Results/OMAM20/OMAM20_{protein_set}/Attention_Matrices"

# === STEP 2: Load and Convert the Lower Triangular P-Distance Matrices ===
def load_lower_triangular_with_labels(file_path):
    """ Reads a labeled lower triangular distance matrix from a CSV file and converts it to a full symmetric matrix. """
    df = pd.read_csv(file_path)
    species = df.iloc[:, 0].values  # Extract species names
    df_values = df.iloc[:, 1:].values.astype(float)  # Convert values to float matrix
    size = len(species)
    full_matrix = np.zeros((size, size))  # Create square matrix

    # Fill lower triangle from CSV
    for i in range(size):
        full_matrix[i, :i+1] = df_values[i, :i+1]
        full_matrix[:i+1, i] = df_values[i, :i+1]  # Mirror upper triangle

    return full_matrix


# Iterate over each protein
for protein in OMAM10_proteins:
    npz_file = f"{npz_folder}/{protein}/{protein}_distance_matrices.npz"
    p_distance_file = f"/home/rohan/ESM/Results/MEGA_Pdist_Matrices/{protein_set}Aligned/{protein}_{protein_set}Aligned_pdist.csv"
    
    data = np.load(npz_file)["arr_0"]  # Extract array if not a dictionary
    p_distance_matrix = load_lower_triangular_with_labels(p_distance_file)
    
    # Ensure the p-distance matrix has the correct shape
    num_layers, num_heads, num_species, _ = data.shape
    assert num_species == p_distance_matrix.shape[0], f"Mismatch in species count for {protein}"
    
    # Flatten upper triangle for correlation (excluding diagonal)
    # Get species names and indices
    species_names = pd.read_csv(p_distance_file).iloc[:, 0].values
    tri_i, tri_j = np.triu_indices_from(p_distance_matrix, k=1)
    
    # Flatten distances and create species pair labels
    p_distance_flat = p_distance_matrix[tri_i, tri_j]
    species_pairs = [f"{species_names[i]} vs {species_names[j]}" for i, j in zip(tri_i, tri_j)]

    
    # === Create a side-by-side figure for Layer 1, Head 1 and Layer 33, Head 20 ===
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #fig.suptitle(f"{protein} - Layer 1, Head 1 vs. Layer 33, Head 20", fontsize=16)

    correlation_series = []
    # Iterate over the two specific layer-head pairs
    for layer in tqdm((31,33), desc="Layers"):
            for head in (19,20):
                attention_matrix = data[layer, head]
                attention_flat = attention_matrix[np.triu_indices_from(attention_matrix, k=1)]
                
                if np.all(attention_flat == attention_flat[0]):
                    pearson_corr, spearman_corr = np.nan, np.nan
                else:
                    pearson_corr, _ = stats.pearsonr(p_distance_flat, attention_flat)
                    spearman_corr, _ = stats.spearmanr(p_distance_flat, attention_flat)
    
                # Show scatter plot
                #plt.figure(figsize=(5, 4))
                plt.figure(figsize=(10,8))
                plt.scatter(p_distance_flat, attention_flat, alpha=0.3, color="dodgerblue", edgecolors='none', s=30)
                
                # Annotate each point
                for x, y, label in zip(p_distance_flat, attention_flat, species_pairs):
                    plt.annotate(
                        label,
                        (x, y),
                        textcoords="offset points",
                        xytext=(3, 3),
                        ha='left',
                        fontsize=7,
                        alpha=0.6
                    )
                
                plt.xlabel("P-Distance", fontsize=12)
                plt.ylabel("ESM-2 Attention Distance", fontsize=12)
                plt.title(f"{protein} L{layer+1} H{head+1}\nPearson: {pearson_corr:.2f}, Spearman: {spearman_corr:.2f}", fontsize=11)
                plt.grid(linestyle="--", alpha=0.7)
                plt.tight_layout()
                plt.show()
        
                # Store correlation for final summary plot
                flat_idx = layer * num_heads + head
                correlation_series.append((layer, head, pearson_corr, spearman_corr))

                
# === FINAL SUMMARY LINE PLOT (Binned by Layer) ===
# Convert to DataFrame
corr_df = pd.DataFrame(correlation_series, columns=["Layer", "Head", "Pearson", "Spearman"])

# Group by layer: average across heads
binned = corr_df.groupby("Layer").mean(numeric_only=True).reset_index()

# Plot
plt.figure(figsize=(14, 6))
plt.plot(binned["Layer"] + 1, binned["Pearson"], label="Pearson", color="dodgerblue", linewidth=2, marker='o')
plt.plot(binned["Layer"] + 1, binned["Spearman"], label="Spearman", color="orange", linewidth=2, marker='s')

plt.title(f"{protein} Correlation with P-Distance (Averaged by Layer)", fontsize=16)
plt.xlabel("Layer", fontsize=14)
plt.ylabel("Mean Correlation", fontsize=14)
plt.xticks(ticks=np.arange(1, binned["Layer"].max()+2), labels=np.arange(1, binned["Layer"].max()+2), rotation=0)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.legend(fontsize=12)
plt.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
