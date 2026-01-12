#!/usr/bin/env python3
"""
Median PC vs. Batch Size — ESM vs. ML (single line plot)

Reads:
- ESM: {protein_set}_PC_results.csv from ESM replicate folder
- ML : ML_PC_results.csv from ML replicate folder

Outputs:
- PNG line plot with two lines (ESM, ML) of median PC by batch size
- Optional CSV of the medians for easy inspection
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

# ==========
# CONFIG
# ==========
protein_set = "OMAM20_34spp"

ESM_RESULTS_CSV = f"/home/rohan/ESM/Results/OMAM20/{protein_set}/Trees/Replicates/{protein_set}_PC_results.csv"
ML_RESULTS_CSV  = "/home/rohan/ESM/Results/OMAM20/ML_MFP/OMAM20_34sppAligned/Replicates/ML_PC_results.csv"

OUTPUT_DIR = os.path.dirname(ESM_RESULTS_CSV)  # you can change if you prefer
PLOT_PNG   = f"/home/rohan/PLM-Phylo/Plots/Fig4_PC_by_batch_size_Linegraph.png"
#MEDIANS_CSV = f"/home/rohan/ESM/Results/OMAM20/{protein_set}_ESM_vs_ML_median_PC_by_batch.csv"

# ==========
# LOAD
# ==========
if not os.path.exists(ESM_RESULTS_CSV):
    raise FileNotFoundError(f"ESM results not found: {ESM_RESULTS_CSV}")
if not os.path.exists(ML_RESULTS_CSV):
    raise FileNotFoundError(f"ML results not found: {ML_RESULTS_CSV}")

esm_df = pd.read_csv(ESM_RESULTS_CSV)   # expects columns: BatchSize, PC
ml_df  = pd.read_csv(ML_RESULTS_CSV)    # expects columns: BatchSize, PC

# Basic sanity checks
for name, df in [("ESM", esm_df), ("ML", ml_df)]:
    if not {"BatchSize", "PC"}.issubset(df.columns):
        raise ValueError(f"{name} CSV must contain columns 'BatchSize' and 'PC'. Got: {df.columns.tolist()}")

# ==========
# AGGREGATE (Median by BatchSize)
# ==========
esm_df["BatchSize"] = esm_df["BatchSize"].round().astype(int)
ml_df["BatchSize"]  = ml_df["BatchSize"].round().astype(int)
esm_med = esm_df.groupby("BatchSize", as_index=False)["PC"].median().rename(columns={"PC": "PC_Median"})
ml_med  = ml_df.groupby("BatchSize",  as_index=False)["PC"].median().rename(columns={"PC": "PC_Median"})

# Join for convenient export (wide format: one col per method)
all_batches = sorted(set(esm_med["BatchSize"]).union(set(ml_med["BatchSize"])))
summary = pd.DataFrame({"BatchSize": all_batches})
summary = summary.merge(esm_med.rename(columns={"PC_Median": "ESM_PC_Median"}),
                        on="BatchSize", how="left")
summary = summary.merge(ml_med.rename(columns={"PC_Median": "ML_PC_Median"}),
                        on="BatchSize", how="left")
#summary.to_csv(MEDIANS_CSV, index=False)

# ==========
# PLOT
# ==========
plt.figure(figsize=(6, 6))

plt.plot(esm_med["BatchSize"], esm_med["PC_Median"],
         marker="o", linewidth=4, markersize=10, label="Median esm-pFM PC", color="dodgerblue")
plt.plot(ml_med["BatchSize"], ml_med["PC_Median"],
         marker="o", linewidth=4, markersize=10, label="Median ML+MFP PC", color="forestgreen")

plt.xlabel("Batch Size (# of proteins)", fontsize=20)
plt.ylabel("Percent Correct (%)", fontsize=20)
plt.ylim(0, 100)

ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))        # <- force integer tick locations
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))  # <- (optional) show no decimals
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

plt.grid(alpha=0.4)
plt.legend(frameon=False, fontsize=20, loc="lower left")
plt.tight_layout()
plt.savefig(PLOT_PNG, dpi=300)
plt.show()
