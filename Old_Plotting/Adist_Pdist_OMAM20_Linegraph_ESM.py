import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm

OMAM20_proteins = [
    "AIMP2", "APOBEC2", "AR", "CASQ1", 
    "COCH", "DYRK1B", "GPX3", "HAUS1", 
    "IL6", "INO80B", "KRT23", "NEUROG1",
    "OPN1SW", "PPOX", "RHO", "SLC39A12", 
    "TF", "TPPP2", "UBA2", "UPRT"
]
protein_set = "4mammals"
#protein_set = "34spp"

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

# Initialize plot
plt.figure(figsize=(7, 6))

for protein in OMAM20_proteins:
    npz_file = f"{npz_folder}/{protein}/{protein}_distance_matrices.npz"
    p_distance_file = f"/home/rohan/ESM/Results/MEGA_Pdist_Matrices/{protein_set}Aligned/{protein}_{protein_set}Aligned_pdist.csv"
    
    data = np.load(npz_file)["arr_0"]  # Extract array if not a dictionary
    p_distance_matrix = load_lower_triangular_with_labels(p_distance_file)
    
    # Ensure the p-distance matrix has the correct shape
    num_layers, num_heads, num_species, _ = data.shape
    assert num_species == p_distance_matrix.shape[0], f"Mismatch in species count for {protein}"
    
    # Flatten upper triangle for correlation (excluding diagonal)
    p_distance_flat = p_distance_matrix[np.triu_indices_from(p_distance_matrix, k=1)]
    
    # === Compute Correlation Across Layers ===
    layer_correlations = []  # Store mean Pearson correlations for each layer

    for layer in tqdm(range(num_layers), desc=f"Processing {protein}"):
        head_correlations = []  # Store Pearson correlations for each head

        for head in range(num_heads):
            attention_matrix = data[layer, head]
            
            # Flatten upper triangular (excluding diagonal)
            attention_flat = attention_matrix[np.triu_indices_from(attention_matrix, k=1)]
            
            # Check if values are valid for correlation
            if np.all(attention_flat == attention_flat[0]):  # Check for constant values
                pearson_corr = np.nan
            else:
                pearson_corr, _ = stats.pearsonr(p_distance_flat, attention_flat)

            head_correlations.append(pearson_corr)

        # Compute mean Pearson correlation for the layer
        mean_correlation = np.nanmean(head_correlations)  # Ignore NaN values in the mean
        layer_correlations.append(mean_correlation)
    
    # === Plot Layer-wise Pearson Correlation ===
    plt.plot(range(num_layers), layer_correlations, marker='.', linestyle='-', label=f"{protein}", alpha=0.6)

# Formatting the graph
plt.xlabel("Layer", fontsize = 18)
num_layers = 33  # or the total number of layers you have

# Set the ticks at positions 0, 5, 10, ..., and label them 1, 6, 11, ...
tick_positions = np.arange(0, num_layers, 3)
tick_labels = np.arange(1, num_layers + 1, 3)

plt.xticks(ticks=tick_positions, labels=tick_labels, fontsize=16)
plt.ylabel("Mean Pearson's R", fontsize = 18)
plt.yticks(fontsize = 16)
plt.title("ESM-2 ADist/Pdist Correlation", fontsize = 20)
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)  # Baseline at 0
#plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
