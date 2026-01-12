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

# === STEP 1: Load Attention Distance Matrices ===
base_msat_folder = "/home/rohan/ESM/Results/OMAM20/MSAT/OMAM20_34sppAligned/Attention_Matrices"

# Initialize plot
plt.figure(figsize=(10, 6))

for protein in tqdm(OMAM20_proteins, desc="Processing Proteins"):
    msat_correlation_file = f"{base_msat_folder}/{protein}/{protein}_layerhead_correlation.csv"
    
    df = pd.read_csv(msat_correlation_file)
    df_sorted = df.sort_values(by=["Layer", "Head"])  # Sort by Layer then Head
    plt.plot(range(len(df_sorted)), df_sorted["R-Squared"].values, label=protein, alpha=0.6)

# Formatting the graph
plt.xlabel("Layer-Head Combination", fontsize=18)
plt.xticks(fontsize=16)
plt.ylabel("Pearson's R", fontsize=18)
plt.yticks(fontsize=16)
plt.title("MSAT/Pdist Correlation for OMAM20 Proteins (144 Points Each)", fontsize=20)
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)  # Baseline at 0
plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid(True)
plt.tight_layout()
plt.show()
