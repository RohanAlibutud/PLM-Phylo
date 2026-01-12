import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import tables and assemble dataframes
df_RF = pd.read_csv("/home/rohan/ESM/Results/OMAM20_RF.csv")
df_CI = pd.read_csv("/home/rohan/ESM/Results/OMAM20_CI.csv")
df_top = df_RF.merge(df_CI, on = "Protein")
df_top["PLM-ss"] = df_top["ESM-2 RF"]
df_top["PLM-msa"] = df_top["MSAT RF"]
df_top["ML+MFP"] = df_top["ML_MFP RF"]

print(df_top)

proteins = df_top["Protein"]
RF_distances = ["ML+MFP", "PLM-ss", "PLM-msa"]
colors = ["dodgerblue", "orange", "forestgreen"]


x = np.arange(len(proteins))  
bar_width = 0.25
fig, ax = plt.subplots(figsize = (18, 6))

ax.grid(axis = "y", alpha = 0.5)
for i, RF_distance in enumerate(RF_distances):
    ax.bar(x + i * bar_width, df_top[RF_distance], width=bar_width, label=RF_distance, color = colors[i])    

ax.set_xticks(x + bar_width)  
ax.tick_params(axis='x', pad=-2)
ax.set_xticklabels(proteins, rotation=45, ha = "center", fontsize = 24)
ax.tick_params(axis='y', labelsize=16)
ax.set_ylabel("Robinson-Foulds\ndistance", fontsize = 24)
ax.set_title("Topological accuracy of PLM-ss, PLM-msa, and ML+MFP by RF distance", fontsize = 30)
ax.legend(fontsize = 16, loc = "upper left", ncol = 3)

plt.tight_layout()
plt.show()

print(df_top["ML+MFP"].mean())
print(df_top["PLM-ss"].mean())
print(df_top["PLM-msa"].mean())

