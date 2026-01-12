import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import tables and assemble dataframes
df_RF = pd.read_csv("/home/rohan/ESM/Results/OMAM20_RF.csv")
df_CI = pd.read_csv("/home/rohan/ESM/Results/OMAM20_CI.csv")
df_top = df_RF.merge(df_CI, on = "Protein")
df_top["PLM-ss"] = df_top["Normalized_CI_Distance_ESM"]
df_top["PLM-msa"] = df_top["Normalized_CI_Distance_MSAT"]
df_top["ML+MFP"] = df_top["Normalized_CI_Distance_ML_MFP"]

print(df_top)

proteins = df_top["Protein"]
CI_distances = ["ML+MFP", "PLM-ss", "PLM-msa"]
colors = ["dodgerblue", "orange", "forestgreen"]


x = np.arange(len(proteins))  
bar_width = 0.25
fig, ax = plt.subplots(figsize = (18, 6))

ax.grid(axis = "y", alpha = 0.5)
for i, CI_distance in enumerate(CI_distances):
    ax.bar(x + i * bar_width, df_top[CI_distance], width=bar_width, label=CI_distance, color = colors[i])    

ax.set_xticks(x + bar_width)  
ax.tick_params(axis='x', pad=-2)
ax.set_xticklabels(proteins, rotation=45, ha = "center", fontsize = 24)
ax.tick_params(axis='y', labelsize=16)
ax.set_ylabel("Normalized Clustering\nInformation distance", fontsize = 24)
ax.set_title("Topological accuracy of PLM-ss, PLM-msa, and ML-MPF by CI distance", fontsize = 30)
ax.legend(fontsize = 16, loc = "upper left", ncol = 3)

plt.tight_layout()
plt.show()

print(df_top["ML+MFP"].mean())
print(df_top["PLM-ss"].mean())
print(df_top["PLM-msa"].mean())
