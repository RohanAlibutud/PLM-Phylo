# LATEST IN USE WAS 2025.12.19

"""
||=========================||
||   ESM-2 Treebuild PSA   ||
||=========================||


python3 /home/rohan/ESM/Scripts/ESM_Treebuilder/ESM_Treebuild_PSA.py \
--model /home/rohan/ESM/Models/ESM2_650M.pt \
--tokenizer /home/rohan/ESM/Models/ESM2_alphabet.pkl \
--input /home/rohan/ESM/Data/Alignments/CFTR/CFTR_100_unaligned.fas \
--out /home/rohan/PLM-Phylo/Results/Trees/Attention_Trees \
--domain CFTR \
--distance cosine \
--pw_align mismatch \
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

# import machine learning libraries
import torch
import pickle

# import phylogenetics packages
from Bio import Phylo
from Bio import SeqIO
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

# import statistical and mathematical packages
import numpy as np

# import file management packages
import os
import io
import argparse
import sys
import psutil
import signal
import gc  
import time
from tqdm import tqdm
import subprocess

# import utils from other ESM scripts
from Pairwise_Aligner import NW_blosum, NW_mismatch
import ESM_Distancer as esm_dist

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
    parser = argparse.ArgumentParser(description = "Run ESM treebuilding pipeline on a specified MSA")
    parser.add_argument("--model", required = True, help = "Path to the pretrained model file")
    parser.add_argument("--tokenizer", required = True, help = "Path to the tokenizer file")
    parser.add_argument("--input", required = True, help = "Path to the list of input sequences")
    parser.add_argument("--out", required = True, help = "Output folder for attention matrices and trees")
    parser.add_argument("--domain", required = True, help = "Domain name used for output naming")
    parser.add_argument("--distance", required = False, help = "Type of distance metric", default = "psa")
    parser.add_argument("--pw_align", required = False, help = "Type of NW alignment [blosum/mismatch]", default = "mismatch")
    parser.add_argument("--save_attn", required = False, action='store_true', help = "Flag to save attention matrices as .npz files")
    parser.add_argument("--astral", required = True, help = "Path to the ASTRAL .jar file")
    args = parser.parse_args()
    
    return args


# METHOD TO RUN TREEBUILDING
def ESM_treebuild(domain, fasta_file, output_folder, tree_folder,
                  model_path, tokenizer_path, distance_metric, 
                  pw_align, save_attn, astral_path):
    print("Running ESM_treebuild() method...")
    
       
    #-------------------------MODEL INITIALIZATION-----------------------------
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting.")
        sys.exit(1)
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    model = torch.load(model_path, weights_only = False)
    tokenizer = pickle.load(open(tokenizer_path, "rb"))
    model.to(device)
    model.eval()
    print(f"Model is on device: {next(model.parameters()).device}")
    
    layer_count = len(model.layers)
    head_count = model.layers[0].self_attn.num_heads
    #--------------------------------------------------------------------------
    
    #----------------------DECLARE DISTANCE MATRIX-----------------------------
    # get count of species
    species_tags = []
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        species_tags.append(record.id)
        sequences.append(str(record.seq))
        
    # dictionary associating species tag with sequence
    spp_seq_dict = {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")}
    
    # declare holding distance matrix, to be populated later
    num_species = len(species_tags)
    distance_matrices = torch.zeros((layer_count, head_count, num_species, num_species), device='cuda') # initialize as a tensor so it can be put on GPU
    #--------------------------------------------------------------------------
    
    # -------------------- DISTANCE METRIC SELECTION --------------------
    if distance_metric == "p_distance":
        distance_fn = esm_dist.p_distance
    elif distance_metric == "euclidean":
        distance_fn = esm_dist.euclidean_distance
    elif distance_metric == "cosine":
        distance_fn = esm_dist.cosine_distance
    elif distance_metric == "frobenius":
        distance_fn = esm_dist.frobenius_distance
    elif distance_metric == "psa_gapless":
        distance_fn = esm_dist.psa_gapless_distance
    elif distance_metric == "pearson":
        distance_fn = esm_dist.pearson_distance
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    # ------------------------------------------------------------------

    #------------------------PAIRWISE COMPARISON-------------------------------
    # for each pair of species, extract sequences, align them, then calculate distances
    print("Starting pairwise distance calculations...")
    for i, sp_A in enumerate(species_tags):
        
        # extract unaligned sequence for species A
        unaligned_seq_A = spp_seq_dict[sp_A]
        
        for j in tqdm(range(i + 1, num_species), desc = f"Calculating distances between {sp_A} and others..."): # skip over redundant comparisons
            sp_B = species_tags[j]
            
            # extract unaligned sequence for species B
            unaligned_seq_B = spp_seq_dict[sp_B]
            
            # align the two sequences via Needleman-Wunsch algorithm
            if pw_align == "blosum":
                aligned_seq_A, aligned_seq_B = NW_blosum(unaligned_seq_A, unaligned_seq_B)
            elif pw_align == "mismatch":
                aligned_seq_A, aligned_seq_B = NW_mismatch(unaligned_seq_A, unaligned_seq_B)
            else:
                print("ERROR: Needleman-Wunsch alignment method not specified")
            
            # EXTRACT ATTENTION MATRICES
            # method will have to be called twice, once for each of the pair
            attn_outdir = os.path.join(output_folder, "Attentions")
            att_A = extract_attn(domain, model, layer_count, head_count, tokenizer, aligned_seq_A, sp_A, save_attn=save_attn, attn_outdir=attn_outdir)
            att_B = extract_attn(domain, model, layer_count, head_count, tokenizer, aligned_seq_B, sp_B,save_attn=save_attn, attn_outdir=attn_outdir)

            # COMPUTE DISSIMILARITY FOR EACH (LAYER, HEAD)
            for layer in range(layer_count):
                for head in range(head_count):
                    # Extract single (seq_len, seq_len) matrix for each layer-head pair
                    matrix_A = att_A[layer, head]
                    matrix_B = att_B[layer, head]
    
                    # compute pairwise distance
                    distance_matrices[layer, head, i, j] = distance_fn(
                        aligned_seq_A, aligned_seq_B, matrix_A, matrix_B
                    )
                    distance_matrices[layer, head, j, i] = distance_matrices[layer, head, i, j]  # Maintain symmetry
    #--------------------------------------------------------------------------
    
    distance_matrices_np = distance_matrices.cpu().numpy()
    np.savez_compressed(f"{output_folder}/{domain}_distance_matrices.npz", distance_matrices_np)

    print("Finished pairwise distance calculations and saved matrices.")

    
    #-------------------BUILD DISTANCE MATRIX AND TREE-------------------------
    # build trees
    tree_building_start_time = time.time()
    build_tree(layer_count, head_count, domain, output_folder, species_tags, distance_matrices, tree_folder)
    domain_tree_folder = f"{tree_folder}/{domain}"
    create_directory(f"{tree_folder}/Consensus_Trees")
    concatenated_newick = f"{tree_folder}/Consensus_Trees/{domain}_concatenated_tree.nwk"
    skip_log = f"{tree_folder}/Consensus_Trees/{domain}_skipped_layerhead_trees.log"
    kept = filter_and_combine_newicks(domain_tree_folder, concatenated_newick, skip_log)
    if kept == 0:
        with open(skip_log, "a") as log_f:
            log_f.write("FATAL\tNo valid layerhead trees remained after filtering; "
                "skipping consensus tree generation.\n")
        print(f"[WARN] {domain}: no valid layerhead trees remained; skipping ASTRAL consensus."        )
        return  # <-- CRITICAL: clean exit, no exception
    call_Astral(astral_path, concatenated_newick, f"{tree_folder}/Consensus_Trees/{domain}_astral_consensus_tree.nwk")
    tree_building_run_time = round(time.time() - tree_building_start_time, 2)
    print(f"Finished the tree building part after {tree_building_run_time/60} minutes")
    #--------------------------------------------------------------------------

# ||=============================||
# ||   DISTANCE MATRIX METHODS   ||
# ||=============================||
def save_attn_npz(attn: np.ndarray, out_npz: str) -> None:
    payload = {}
    LAY, H, _, _ = attn.shape
    for li in range(LAY):
        for hi in range(H):
            payload[f"layer{li+1:02d}_head{hi+1:02d}"] = attn[li, hi]
    np.savez_compressed(out_npz, **payload)

def extract_attn(domain, model, layer_count, head_count, alphabet, protein_sequence, sequence_identifier,
                 save_attn=False, attn_outdir=None):
    import os
    import numpy as np
    import torch

    # --------------------------
    # Tokenize
    # --------------------------
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([("species_id", protein_sequence)])

    # Assumption: one sequence per call
    assert batch_tokens.shape[0] == 1, f"Expected batch=1, got {batch_tokens.shape}"

    device = next(model.parameters()).device
    batch_tokens = batch_tokens.to(device)

    L = len(protein_sequence)
    T = int(batch_tokens.shape[-1])

    # --------------------------
    # Forward pass
    # --------------------------
    model.eval()
    with torch.no_grad():
        out = model(batch_tokens, need_head_weights=True)
        attn = out["attentions"].detach()

    # Assumption: attentions are 5D
    assert attn.ndim == 5, f"Expected attentions ndim=5, got {attn.ndim}, shape={tuple(attn.shape)}"

    # --------------------------
    # Normalize to (layers, heads, T, T)
    # --------------------------
    if attn.shape[0] == layer_count:
        # (layers, batch, heads, T, T)
        assert attn.shape[1] == 1, f"Expected batch=1, got shape={tuple(attn.shape)}"
        assert attn.shape[2] == head_count, f"Expected heads={head_count}, got shape={tuple(attn.shape)}"
        attn = attn[:, 0, :, :, :]  # (layers, heads, T, T)
    else:
        # (batch, layers, heads, T, T)
        assert attn.shape[0] == 1, f"Expected batch=1, got shape={tuple(attn.shape)}"
        assert attn.shape[1] == layer_count, f"Expected layers={layer_count}, got shape={tuple(attn.shape)}"
        assert attn.shape[2] == head_count, f"Expected heads={head_count}, got shape={tuple(attn.shape)}"
        attn = attn[0, :, :, :, :]  # (layers, heads, T, T)

    # --------------------------
    # Slice AA tokens only (drop BOS/EOS)
    # --------------------------
    start = 1
    end = start + L
    assert end <= T, f"Slicing out of bounds: L={L}, T={T}, end={end}"

    attn = attn[:, :, start:end, start:end]  # (layers, heads, L, L)
    assert attn.shape[-1] == L and attn.shape[-2] == L, f"Post-slice shape wrong: {tuple(attn.shape)}"

    # Cast to float16 numpy
    attn_np = attn.to("cpu").to(torch.float16).numpy()

    # --------------------------
    # Optional save
    # --------------------------
    if save_attn:
        assert attn_outdir is not None, "save_attn=True but attn_outdir is None"
        os.makedirs(attn_outdir, exist_ok=True)
        out_npz = os.path.join(attn_outdir, f"{domain}_{sequence_identifier}_attentions.npz")

        payload = {}
        for li in range(layer_count):
            for hi in range(head_count):
                payload[f"layer{li+1:02d}_head{hi+1:02d}"] = attn_np[li, hi]

        np.savez_compressed(out_npz, **payload)

    return attn_np


    
# ||=======================||
# ||   PHYLOGENY METHODS   ||
# ||=======================||

# METHOD TO BUILD NEIGHBOR-JOINING TREE FROM DISTANCE MATRIX
def build_tree(layer_count, head_count, domain, output_folder, species_tag, distance_matrices, tree_folder):
    for l in range(layer_count):
        for h in range(head_count):
            chosen_layer = l + 1
            chosen_head = h + 1
            print(f"chosen_layer = {chosen_layer}, chosen_head = {chosen_head}")
            species_matrix = distance_matrices[l, h, :, :]
            create_directory(f"{tree_folder}/{domain}/layer{chosen_layer}")
            matrix_path = f"{tree_folder}/{domain}/layer{chosen_layer}/{domain}_layer{chosen_layer}_head{chosen_head}_distance_matrix.csv"
            tree_path = f"{tree_folder}/{domain}/layer{chosen_layer}/{domain}_layer{chosen_layer}_head{chosen_head}_tree.nwk"
            np.savetxt(matrix_path, species_matrix.cpu().numpy(), delimiter=",")
            species_matrix_list = species_matrix.tolist()
            lower_triangle_matrix = []
            for i in range(len(species_matrix_list)):
                row = []
                for j in range(len(species_matrix_list[i])):
                    if j <= i:
                        row.append(species_matrix_list[i][j])
                gc.collect()
                lower_triangle_matrix.append(row)
            dm = DistanceMatrix(names=species_tag, matrix=lower_triangle_matrix)
            constructor = DistanceTreeConstructor()
            nj_tree = constructor.nj(dm)
            Phylo.write(nj_tree, tree_path, "newick")
            print(f"Saved tree to {tree_path}")

# SUBMETHOD TO TEST TREES FOR PROBLEMS
def tree_has_nonpositive_branches(tree_path: str) -> tuple[bool, str]:
    """
    Returns (is_bad, reason). "bad" means ANY branch length is <= 0.
    """
    try:
        tree = Phylo.read(tree_path, "newick")
    except Exception as e:
        return True, f"parse_error: {e}"

    for clade in tree.find_clades():
        bl = getattr(clade, "branch_length", None)
        if bl is None:
            continue  # unlabeled branch length is allowed
        try:
            bl_val = float(bl)
        except Exception:
            return True, f"non_numeric_branch_length: {bl!r}"

        if bl_val <= 0.0:
            return True, f"nonpositive_branch_length: {bl_val}"

    return False, "ok"

# SUBMETHOD TO COMBINE NEWICK TREES AFTER FILTERING
def filter_and_combine_newicks(input_folder_path: str, output_newick: str, log_path: str) -> int:
    """
    Writes only "good" trees (no nonpositive branch lengths) into output_newick.
    Appends skipped trees to log_path. Returns number of kept trees.
    """
    kept = 0
    skipped = 0

    os.makedirs(os.path.dirname(output_newick), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(output_newick, "w") as out_f, open(log_path, "a") as log_f:
        log_f.write(f"\n=== Filtering run: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_f.write(f"Input folder: {input_folder_path}\n")
        log_f.write(f"Output newick: {output_newick}\n\n")

        for root, _, files in os.walk(input_folder_path):
            for filename in files:
                if not filename.endswith(".nwk"):
                    continue
                tree_path = os.path.join(root, filename)

                is_bad, reason = tree_has_nonpositive_branches(tree_path)
                if is_bad:
                    skipped += 1
                    log_f.write(f"SKIP\t{tree_path}\t{reason}\n")
                    continue

                with open(tree_path, "r") as in_f:
                    tree_str = in_f.read().strip()
                if not tree_str:
                    skipped += 1
                    log_f.write(f"SKIP\t{tree_path}\tempty_file\n")
                    continue

                out_f.write(tree_str + "\n")
                kept += 1

        log_f.write(f"\nSummary: kept={kept}, skipped={skipped}\n")

    print(f"[filter_and_combine_newicks] kept={kept}, skipped={skipped}")
    print(f"[filter_and_combine_newicks] log: {log_path}")
    return kept

# SUBMETHOD TO PREPARE NJ TREES FOR ASTRAL CONSENSUS
def combine_newicks(input_folder_path, output_newick):
    print(f"Concatenating Newick trees in folder: {input_folder_path}")
    with open(output_newick, 'w') as output_file:
        for root, dirs, files in os.walk(input_folder_path):
            for filename in files:
                if filename.endswith('.nwk'):  
                    file_path = os.path.join(root, filename)
                    with open(file_path, 'r') as input_file:
                        tree = input_file.read().strip()
                        output_file.write(tree + '\n')


# BUILD CONSENSUS TREE
def call_Astral(astral_path, input_newick, output_newick):
    subprocess.call([
        'java', '-jar', astral_path,
        '-i', input_newick,
        '-o', output_newick,
        '-t', '1'
    ])


# ||========================||
# ||   UTILITY SUBMETHODS   ||
# ||========================||

# SUBMETHOD TO ACQUIRE RECORD INFORMATION FROM FASTA
def get_species_tag_and_sequence(fasta_file):
    species_tags = []
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        species_tags.append(record.id)
        sequences.append(str(record.seq))
    return species_tags, sequences

# SUBMETHOD TO CREATE A DIRECTORY
def create_directory(path):
    os.makedirs(path, exist_ok=True)

# SUBMETHOD TO CHECK FOR SUFFICIENT STORAGE SPACE
def check_disk_space():
    disk_usage = psutil.disk_usage('/')
    available_percentage = (disk_usage.free / disk_usage.total) * 100
    if available_percentage < 10:
        print("ALERT: Available disk space is less than 10%. Stopping all Python scripts.")
        
        # hard abort if it looks like we're going to fill things up
        for proc in psutil.process_iter(attrs=['pid', 'name']):
            if 'python' in proc.info['name'].lower():
                try:
                    os.kill(proc.info['pid'], signal.SIGTERM)
                    print(f"Terminated process ID {proc.info['pid']} ({proc.info['name']})")
                except (psutil.NoSuchProcess, PermissionError) as e:
                    print(f"Error terminating process ID {proc.info['pid']}: {e}")



"""
╔═══╗╔═══╗╔═╗╔═╗╔═╗╔═╗╔═══╗╔═╗ ╔╗╔═══╗╔═══╗
║╔═╗║║╔═╗║║║╚╝║║║║╚╝║║║╔═╗║║║╚╗║║╚╗╔╗║║╔═╗║
║║ ╚╝║║ ║║║╔╗╔╗║║╔╗╔╗║║║ ║║║╔╗╚╝║ ║║║║║╚══╗
║║ ╔╗║║ ║║║║║║║║║║║║║║║╚═╝║║║╚╗║║ ║║║║╚══╗║
║╚═╝║║╚═╝║║║║║║║║║║║║║║╔═╗║║║ ║║║╔╝╚╝║║╚═╝║
╚═══╝╚═══╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝ ╚╝╚╝ ╚═╝╚═══╝╚═══╝
"""

# ||==========================================||
# ||    ESM TREEBUILD FROM FULL SINGLE MSA    ||
# ||==========================================||
if __name__ == "__main__":
    args = parse_arguments()
    base_output_folder = os.path.join(args.out, "Attention_Matrices", args.domain)
    create_directory(base_output_folder)
    tree_folder = os.path.join(args.out, "Trees", args.domain)
    
    start_time = time.time()

    ESM_treebuild(
        args.domain,
        args.input,
        base_output_folder,
        tree_folder,
        args.model,
        args.tokenizer,
        args.distance,
        args.pw_align,
        args.save_attn,
        args.astral
    )

    
    end_time = time.time()
    run_time = round(end_time - start_time, 2)
    print(f"ESM_Treebuilder finished after {run_time} seconds.\n\n")

