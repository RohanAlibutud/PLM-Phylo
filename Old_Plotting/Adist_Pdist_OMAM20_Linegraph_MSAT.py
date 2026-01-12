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
#protein_set = "4mammals"
protein_set = "34spp"

# === STEP 1: Load Attention Distance Matrices ===
base_msat_folder = f"/home/rohan/ESM/Results/OMAM20/MSAT/OMAM20_{protein_set}Aligned/Attention_Matrices"

# Initialize plot
plt.figure(figsize=(7, 6))

for protein in tqdm(OMAM20_proteins, desc="Processing Proteins"):
    msat_correlation_file = f"{base_msat_folder}/{protein}/{protein}_layerhead_correlation.csv"
    
    df = pd.read_csv(msat_correlation_file)
    mean_correlation = df.groupby("Layer")["R-Squared"].mean()
    plt.plot(mean_correlation.index, mean_correlation.values, label=protein, marker = ".", alpha=0.6)
    

# Formatting the graph
plt.xlabel("Layer", fontsize = 18)
plt.xticks(fontsize = 16)
plt.ylabel("Mean Pearson's R", fontsize = 18)
plt.yticks(fontsize = 16)
plt.title("MSAT ADist/Pdist Correlation", fontsize = 20)
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)  # Baseline at 0
#plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

