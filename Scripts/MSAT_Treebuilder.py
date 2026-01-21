"""
MSAT_Treebuilder.py
Last updated: 2025.12.19
> Designed to work on Korolev
> CUDA-mandatory
> Loads model in from pretrained library, not saved file

TEMPLATE COMMAND: 
python3 /home/rohan/ESM/Scripts/MSA_Transformer/MSAT_Treebuilder.py \
  --msa /home/rohan/PLM-Phylo/Data/Alignments/OMAM20/OMAM20_4mammalsAligned/TF_4mammalsAligned.fas \
  --out /home/rohan/PLM-Phylo/Results/Trees/MSAT_Trees/OMAM20_4mammals/ \
  --domain TF_4mammals \
  --astral /home/rohan/Tools/Astral.5.7.8/Astral/astral.5.7.8.jar

"""



"""
╔╗   ╔══╗╔══╗ ╔═══╗╔═══╗╔═══╗╔══╗╔═══╗╔═══╗
║║   ╚╣╠╝║╔╗║ ║╔═╗║║╔═╗║║╔═╗║╚╣╠╝║╔══╝║╔═╗║
║║    ║║ ║╚╝╚╗║╚═╝║║║ ║║║╚═╝║ ║║ ║╚══╗║╚══╗
║║ ╔╗ ║║ ║╔═╗║║╔╗╔╝║╚═╝║║╔╗╔╝ ║║ ║╔══╝╚══╗║
║╚═╝║╔╣╠╗║╚═╝║║║║╚╗║╔═╗║║║║╚╗╔╣╠╗║╚══╗║╚═╝║
╚═══╝╚══╝╚═══╝╚╝╚═╝╚╝ ╚╝╚╝╚═╝╚══╝╚═══╝╚═══╝                                                                                 
"""

# import machine learnign libraries
import torch
import pickle
from esm import pretrained


# import phylogenetics libraries
import dendropy
from Bio import Phylo
from Bio import SeqIO
from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import DistanceCalculator

# import data and statistics libraries
import csv
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# import utility libraries
import os
import re
import io
import sys
import time
import string
import psutil
import shutil
import argparse
import itertools
import subprocess
from typing import List, Tuple
from tqdm import tqdm

# import custom libraries
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from RF_scorer import rf_distance
from ASTRAL_Calls import combine_newicks, call_Astral

"""
╔═╗╔═╗╔═══╗╔════╗╔╗ ╔╗╔═══╗╔═══╗╔═══╗
║║╚╝║║║╔══╝║╔╗╔╗║║║ ║║║╔═╗║╚╗╔╗║║╔═╗║
║╔╗╔╗║║╚══╗╚╝║║╚╝║╚═╝║║║ ║║ ║║║║║╚══╗
║║║║║║║╔══╝  ║║  ║╔═╗║║║ ║║ ║║║║╚══╗║
║║║║║║║╚══╗ ╔╝╚╗ ║║ ║║║╚═╝║╔╝╚╝║║╚═╝║
╚╝╚╝╚╝╚═══╝ ╚══╝ ╚╝ ╚╝╚═══╝╚═══╝╚═══╝
"""

# ||==================||
# ||   MAIN METHODS   ||
# ||==================||

# METHOD TO PARSE ARGUMENTS
def parse_arguments():
    parser = argparse.ArgumentParser(description = "Run MSAT treebuilding pipeline on a specified MSA")
    #parser.add_argument("--model", required = True, help = "Path to the pretrained model file")
    #parser.add_argument("--tokenizer", required = True, help = "Path to the tokenizer file")
    parser.add_argument("--msa", required = True, help = "Path to the MSA file (FASTA format)")
    parser.add_argument("--out", required = True, help = "Output folder for attention matrices and trees")
    parser.add_argument("--domain", required = True, help = "Domain name used for output naming")
    parser.add_argument("--astral_path", required = True, help = "Path to the ASTRAL jar file for consensus tree generation")
    parser.add_argument("--rows_wanted", required = False, default = False, help = "Output row attention matrices as well?")
    args = parser.parse_args()

    return args

# METHOD TO INFER PHYLOGENETIC TREE FROM ALIGNMENT
#def MSAT_treebuild(domain, fasta_file, output_folder, tree_folder, model_path, tokenizer_path, astral_path):
def MSAT_treebuild(domain, fasta_file, output_folder, tree_folder, astral_path, rows_wanted):
    print("Running MSAT_treebuild()...")

    # initialize model
    model, alphabet, device = init_from_library()

    # extract attentions
    batch_converter = alphabet.get_batch_converter()
    msa_data = get_species_tag_and_sequence(fasta_file)
    msa_batch_labels, msa_batch_strs, msa_batch_tokens = batch_converter(msa_data)
    msa_batch_tokens = msa_batch_tokens.to(device)
    with torch.no_grad():
        results = model(msa_batch_tokens, repr_layers=[12], need_head_weights=True)
    row_attns = results["row_attentions"].cpu().numpy()[0, :, :, 1:, 1:]
    col_attns = results["col_attentions"].cpu().numpy()[0, :, :, 1:, :, :]
    print(results["col_attentions"].shape)
    
    # Generate matrices
    generate_matrices(domain, output_folder, row_attns, col_attns, rows_wanted)

    # build layerhead trees
    matrix_folder = f"{output_folder}/{domain}_Matrices/Meanmatrices"
    tree_folder = f"{output_folder}/Trees"
    layerhead_att_vs_p(output_folder, domain, fasta_file, matrix_folder, tree_folder)

    # perform consensus treebuilding
    consensus_start_time = time.time()
    aDist_tree_folder = f"{tree_folder}/aDist_Trees"
    master_newick_path = f"{output_folder}/master_newick.nwk"
    astral_output_path = f"{output_folder}/{domain}_astral_output.nwk"
    combine_newicks(aDist_tree_folder, master_newick_path)  # Combine trees into one Newick file
    call_Astral(master_newick_path, astral_output_path, astral_path)
    consensus_run_time = round(time.time() - consensus_start_time, 2)


# METHOD TO INITIALIZE MODEL FROM LIBRARY
def init_from_library():
    print("Initializing model...")

    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting.")
        sys.exit(1)
    device = torch.device('cuda')
    model, alphabet = pretrained.esm_msa1b_t12_100M_UR50S()
    model.to(device)
    model.eval()

    print(f"Model initialized on device: {device}")
    return model, alphabet, device

# METHOD TO INITIALIZE MODEL FROM FILE
def init_from_file(model_path, tokenizer_path):
    print("Initializing model...")

    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting.")
        sys.exit(1)
    device = torch.device('cuda')

    model = torch.load(model_path)
    tokenizer = pickle.load(open(tokenizer_path, "rb"))
    model.to(device)
    model.eval()

    print(f"Model initialized on device: {device}")
    return model, tokenizer, device

# METHOD TO GENERATE ATTENTION MATRICES
def generate_matrices(domain, output_folder, row_attns, col_attns, rows_wanted):
    if rows_wanted:
        print(f"Processing row matrices for {domain}...")
        create_directory(f"{output_folder}/{domain}_Matrices/Rowmatrices/")
        for x in range(len(row_attns)):
            for y in tqdm(range(len(row_attns[0])), desc=f"Layer {x+1}"):
                layer_tag = f"layer{x+1}"
                head_tag = f"head{y+1}"
                filename_string = f"{layer_tag}_{head_tag}"
                row_matrix = row_attns[x][y]
                np.savetxt(f"{output_folder}/{domain}_Matrices/Rowmatrices/{domain}_{filename_string}_rowmatrix.csv", row_matrix, delimiter=",")
    
    print(f"Processing column matrices for {domain}...")
    for x in range(len(col_attns)):
        print(col_attns.shape)
        for y in tqdm(range(len(col_attns[0])), desc=f"Layer {x+1}"):
            layer_tag = f"layer{x+1}"
            head_tag = f"head{y+1}"
            all_matrices = []
            for z in range(len(col_attns[0][0])):
                residue_tag = f"residue{z+1}"
                filename_string = f"{layer_tag}_{head_tag}_{residue_tag}"
                col_matrix = col_attns[x][y][z]
                all_matrices.append(col_attns[x][y][z])
                create_directory(f"{output_folder}/{domain}_Matrices/Colmatrices/{residue_tag}")
                np.savetxt(f"{output_folder}/{domain}_Matrices/Colmatrices/{residue_tag}/{domain}_{filename_string}_colmatrix.csv", col_matrix, delimiter=",")
            mean_matrix = np.mean(all_matrices, axis=0)
            filename_string = f"{layer_tag}_{head_tag}"
            create_directory(f"{output_folder}/{domain}_Matrices/Meanmatrices")
            np.savetxt(f"{output_folder}/{domain}_Matrices/Meanmatrices/{domain}_{filename_string}_meanmatrix.csv", mean_matrix, delimiter=",")

# METHOD TO ACQUIRE RECORD INFORMATION FROM FASTA
def get_species_tag_and_sequence(fasta_file):
    msa_data = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        msa_data.append((record.id, str(record.seq)))  # Make it a tuple
    return msa_data


# METHOD TO CREATE DIRECTORY
def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        print(f"Creation of the directory '{path}' failed: {error}")




# ||============================||
# ||   LAYERHEAD TREEBUILDING   ||
# ||============================||

# METHOD TO COMPARE SPECIES ATTENTION VALUES AND P-DISTANCE AND BUILD TREES
def layerhead_att_vs_p(output_base_folder, domain, domain_MSA, meanmatrix_folder, tree_folder):
    print("Generating layerhead species attention vs. p-distance plot...")
    
    # import p-distance matrix from MSA
    pdist_matrix = np.array(p_distance_calc(domain_MSA))
    
    # output p_dist matrix just to check
    with open(f"{output_base_folder}/" + domain + "_p_distance_matrix.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(pdist_matrix)
        
    # EXTRACT DATA FROM EACH LAYERHEAD MATRIX OBJECT
    layerhead_matrices = []
    r_squared_values = [] 
    correlation_table = [] 
    for matrix_file in tqdm(os.listdir(meanmatrix_folder)):
        file_path = os.path.join(meanmatrix_folder, matrix_file)
    
        # create an object for each layerhead matrix 
        layer, head = parse_layerhead(matrix_file)
        if layer is None or head is None:
            print(f"Skipping file {matrix_file} as it doesn't contain layer or head information.")
            continue
        matrix = pd.read_csv(file_path, header=None).astype(float)
        lh_matrix_obj = lh_matrix(np.array(matrix), layer, head)
        layerhead_matrices.append(lh_matrix_obj)      
    
        # calculate correlation against p-distance 
        flattened_matrix_data = lh_matrix_obj.data.flatten()
        flattened_pdist_data = pdist_matrix.flatten()
        try:
            correlation, _ = pearsonr(flattened_matrix_data, flattened_pdist_data)
            r_squared = correlation ** 2
            r_squared_values.append(r_squared)
            
            # Assign R-squared value to lh_matrix object
            lh_matrix_obj.r_squared = r_squared
            correlation_table.append([layer, head, r_squared])
    
        except ValueError as e:
            print("Error calculating correlation for Layer:", layer, "Head:", head)
            print("Error message:", e)
            print("Length of flattened pdist_matrix:", len(pdist_matrix))
            print("Length of flattened lh_matrix data:", len(flattened_matrix_data))
    
    # export correlation table to .csv file
    correlation_output_path = os.path.join(output_base_folder, f"{domain}_layerhead_correlation.csv")
    with open(correlation_output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Layer", "Head", "R-Squared"])  # Write header
        writer.writerows(correlation_table)

    print(f"Correlation table saved to {correlation_output_path}")

    # ADD ATTENTION DISTANCE AND BUILD TREES
    for lh_matrix_obj in tqdm(layerhead_matrices, desc = "Calculating attention distances"):
        # calculate attention distance
        calc_aDist(lh_matrix_obj)
        
        # build tree from attention distance matrix using BioPython DistanceMatrix tree builder
        create_directory(f"{tree_folder}/aDist_Trees/")
        tree_outpath = f"{tree_folder}/aDist_Trees/{domain}_layer{lh_matrix_obj.layer}_head{lh_matrix_obj.head}_aDist_tree.nwk"
        aDist_matrix = lh_matrix_obj.aDist
        build_tree(domain, domain_MSA, aDist_matrix, tree_outpath)

# OBJECT FOR LAYERHEAD MATRIX
class lh_matrix:
    def __init__(self, data, layer, head):
        self.data = data
        self.layer = layer
        self.head = head
        self.pdist_rsquared = None

# METHOD TO BUILD TREE FROM ATTENTION DISTANCE MATRIX
def build_tree(protein_name, msa, input_matrix, tree_outpath):
    species_list = [record.id for record in SeqIO.parse(msa, "fasta")]
    if type(input_matrix) == np.ndarray:
        matrix_list = input_matrix.tolist()
    else:
        matrix_list = input_matrix
    for i, species in enumerate(species_list):
        matrix_list[i].insert(0, species)
    species_list.insert(0, '')
    matrix_list.insert(0, species_list)
    lower_triangle_matrix = [matrix_list[i][:i+1] for i in range(len(matrix_list))]
    dm = DistanceMatrix(species_list[1:], [row[1:] for row in lower_triangle_matrix[1:]])
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)
    with open(tree_outpath, "w") as f:
        Phylo.write(tree, f, "newick")

# SUBMETHOD TO PARSE LAYERS AND HEADS FROM FILENAME    
def parse_layerhead(filename):
    layer_match = re.search(r'layer(\d+)', filename)
    head_match = re.search(r'head(\d+)', filename)
    layer = int(layer_match.group(1)) if layer_match else None
    head = int(head_match.group(1)) if head_match else None
    return layer, head

# SUBMETHOD TO FIND P-DISTANCE FROM EACH MSA
def p_distance_calc(fasta_file):
    print("Calculating p-distance for " + fasta_file)
    
    # import MSA
    MSA = AlignIO.read(fasta_file, "fasta")
    
    # call BioPhylo's distance calculator function
    calculator = DistanceCalculator('identity')
    pdist_matrix = calculator.get_distance(MSA)

    print("Finished p-distance calculations")    
    return pdist_matrix

# SUBMETHOD TO CALCULATE ATTENTION DISTANCE
def calc_aDist(lh_matrix_obj):
    # find maximum species attention in this layerhead matrix for normalization
    att_matrix = lh_matrix_obj.data # get attention value matrix
    max_att = np.max(att_matrix)
    
    # subtract by pairwise attention for each element in matrix and apply values to new matrix
    aDist_matrix = max_att - att_matrix
    
    # return new matrix as part of lh_matrix_obj
    lh_matrix_obj.aDist = aDist_matrix
    return lh_matrix_obj



"""
╔═══╗╔═══╗╔═╗╔═╗╔═╗╔═╗╔═══╗╔═╗ ╔╗╔═══╗╔═══╗
║╔═╗║║╔═╗║║║╚╝║║║║╚╝║║║╔═╗║║║╚╗║║╚╗╔╗║║╔═╗║
║║ ╚╝║║ ║║║╔╗╔╗║║╔╗╔╗║║║ ║║║╔╗╚╝║ ║║║║║╚══╗
║║ ╔╗║║ ║║║║║║║║║║║║║║║╚═╝║║║╚╗║║ ║║║║╚══╗║
║╚═╝║║╚═╝║║║║║║║║║║║║║║╔═╗║║║ ║║║╔╝╚╝║║╚═╝║
╚═══╝╚═══╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝ ╚╝╚╝ ╚═╝╚═══╝╚═══╝
"""

# ||==========================================||
# ||    MSAT TREEBUILD FROM FULL SINGLE MSA    ||
# ||==========================================||
if __name__ == "__main__":
    args = parse_arguments()
    
    print("DEBUG args.domain =", args.domain)
    print("DEBUG args.msa    =", args.msa)
    print("DEBUG args.out    =", args.out)


    base_output_folder = os.path.join(args.out, args.domain)
    create_directory(base_output_folder)
    tree_folder = os.path.join(args.out, "Trees", args.domain)
    
    start_time = time.time()

    MSAT_treebuild(
        args.domain,
        args.msa,
        base_output_folder,
        tree_folder,
        #args.model,
        #args.tokenizer,
        args.astral_path,
        args.rows_wanted,
        #args.extraction_on,
        #args.gapskip_on
    )
    
    end_time = time.time()
    run_time = round(end_time - start_time, 2)
    print(f"MSAT_Treebuilder finished after {run_time} seconds.")