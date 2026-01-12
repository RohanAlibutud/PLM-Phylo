import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# METHOD TO CALCULATE PERCENTAGE CORRECT OF A TREE
def calc_PC(Percentage, num_species = 34):
    RF = Percentage
    N = num_species
    
    PIC = (RF/2) / (N-3)
    PC = round( ((1 - PIC) * 100), 1)
    
    return PC


# ANALYSIS

# import tables and assemble dataframes
df_RF = pd.read_csv("/home/rohan/ESM/Results/OMAM20_RF.csv")
proteins = df_RF["Protein"]

print(df_RF)

df_RF["PLM-ss"] = df_RF["ESM-2 RF"].apply(calc_PC)
df_RF["PLM-msa"] = df_RF["MSAT RF"].apply(calc_PC)
df_RF["ML+MFP"] = df_RF["ML_MFP RF"].apply(calc_PC)
Percentages = ["PLM-ss", "ML+MFP"]
colors = ["dodgerblue", "green"]

x = np.arange(len(proteins))  
bar_width = 0.25
gap = 0.05

fig, ax = plt.subplots(figsize=(18, 6))

ax.grid(axis="y", alpha=0.5)

# Left bar: PLM-ss, Right bar: ML+MFP

ax.bar(x - bar_width - gap, df_RF["PLM-msa"], width=bar_width, label="PLM-msa", color="orange")
ax.bar(x, df_RF["PLM-ss"], width=bar_width, label="PLM-ss", color="dodgerblue")
ax.bar(x + bar_width + gap, df_RF["ML+MFP"], width=bar_width, label="ML+MFP", color="green")

# Labels & Ticks
ax.set_xticks(x)
ax.set_xticklabels(proteins, rotation=70, ha="center", fontsize=24)
ax.set_ylabel("Percent Correct (%)", fontsize=24)
ax.set_ylim(0, 110)  # Headroom for 100%
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='x', pad=-2)
ax.set_title("Percentage of correctly inferred clades (PC), PLM-ss vs. ML+MFP", fontsize=30)
ax.legend(fontsize=16, loc="upper left", ncol=2)

plt.tight_layout()
plt.show()

ML_median_PC = df_RF["ML+MFP"].median()
SS_median_PC = df_RF["PLM-ss"].median()
MSA_median_PC = df_RF["PLM-msa"].median()

print(f"ML+MFP median PC: {ML_median_PC}")
print(f"PLM-ss median PC: {SS_median_PC}")
print(f"PLM-msa median PC: {MSA_median_PC}")

# Count how many proteins have PLM-ss within ±5% of ML+MFP
df_RF["Comparable"] = (abs(df_RF["PLM-ss"] - df_RF["ML+MFP"]) <= 20)

num_comparable = df_RF["Comparable"].sum()
total_proteins = len(df_RF)
percentage_comparable = (num_comparable / total_proteins) * 100

print(f"\nProteins where PLM-ss is within ±5% of ML+MFP: {num_comparable}/{total_proteins} ({percentage_comparable:.1f}%)")




# ||==============================||
# ||   HISTOGRAM OF DIFFERENCES   ||
# ||==============================||

"""
from scipy.stats import ttest_rel, wilcoxon

# Paired t-test
t_stat, p_val_ttest = ttest_rel(df_RF["PLM-ss"], df_RF["ML+MFP"])
print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val_ttest:.4f}")

# Wilcoxon signed-rank test (non-parametric)
w_stat, p_val_wilcoxon = wilcoxon(df_RF["PLM-ss"], df_RF["ML+MFP"])
print(f"Wilcoxon test: W = {w_stat}, p = {p_val_wilcoxon:.4f}")

df_RF["Diff"] = df_RF["PLM-ss"] - df_RF["ML+MFP"]

# Optional: histogram of differences
df_RF["Diff"].hist(bins=10, edgecolor='black')
plt.axvline(0, color='red', linestyle='--')
plt.title("Difference (PLM-ss - ML+MFP)")
plt.xlabel("PC Difference (%)")
plt.ylabel("Protein Count")
plt.show()

"""