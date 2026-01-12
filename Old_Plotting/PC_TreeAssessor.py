import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# METHOD TO CALCULATE PERCENTAGE CORRECT OF A TREE
def calc_PC(RF_distance, num_species = 34):
    RF = RF_distance
    N = num_species
    
    PIC = (RF/2) / (N-3)
    PC = round( ((1 - PIC) * 100), 1)
    
    return PC

# import tables and assemble dataframes
df = pd.read_csv("/home/rohan/ESM/Results/OMAM20_RF.csv", index_col=False)
df.reset_index(drop=True, inplace=True)

df["PLM-ss"] = df["ESM-2 RF"].apply(calc_PC)
df["PLM-msa"] = df["MSAT RF"].apply(calc_PC)
df["ML+MFP"] = df["ML_MFP RF"].apply(calc_PC)
print(df)
 

mean_PC_ESM = df["PLM-ss"].mean()
mean_PC_MSAT = df["PLM-msa"].mean()
mean_PC_ML = df["ML+MFP"].mean()

print(f"mean percentage correct, PLM-ss: {mean_PC_ESM:.1f}")
print(f"mean percentage correct, PLM-msa: {mean_PC_MSAT:.1f}")
print(f"mean percentage correct, ML+MFP: {mean_PC_ML:.1f}")

median_PC_ESM = df["PLM-ss"].median()
median_PC_MSAT = df["PLM-msa"].median()
median_PC_ML = df["ML+MFP"].median()

print(f"median percentage correct, PLM-ss: {median_PC_ESM}")
print(f"median percentage correct, PLM-msa: {median_PC_MSAT}")
print(f"median percentage correct, ML+MFP: {median_PC_ML}")

