# PLM-Phylo
This is a repository for the scripts used in the paper "Multiple versus pairwise sequence alignments for protein phylogenetics using deep learning models", submitted to ISMB 2026 as a conference proceeding. This paper introduced two methods for inferring phylogenetic trees from the attention matrices of protein foundation models, specifically, MSA Transformer and ESM-2.

### MSA Transformer sequence attention matrix phylogenetic inference
![msa-pFM treebuilding](Figures/Fig1_msat-pFM_treebuilding.png)
- (a) A target sequence must be used to compose a multiple sequence alignment alongside all matching BLAST hits
- (b) The resulting MSA (c) is then passed into the pretrained msa-pFM
- (d) Model inference then yields a set of 144 SAMs of dimensions N x N x L (e), where N is the number of sequences in the input MSA and L is the sequence length
- (f) Each SAM is averaged across L to derive a single SAM of dimensions N x N, which is then inverted to create a distance matrix representing the predicted evolutionary distance between any two sequences
- (g) The distance matrix is passed to the neighbor-joining algorithm to infer a phylogenetic tree. This process yields a single tree for every attention layer/attention head combination
- Multiple trees are combined using ASTRAL in order to yield a final consensus tree


### ESM-2 residue attention matrix phylogenetic inference
![esm-pFM treebuilding](Figures/Fig2_esm-pFM_treebuilding.png)
 - (a) Two or more unaligned sequences are passed into the pretrained esm-pFM model
 - (b) Model inference yields (c) N attention matrices of dimensions L x L, where N is the number of sequences and L is the length of each sequence
 - Distance metrics are calculated for pairwise comparisons, producing a distance matrix (d) that represents the dissimilarity between the attentions of any two species
 - The distance matrix is passed to the neighbor-joining algorithm to infer a phylogenetic tree (e) This process yields a single tree for every attention layer/attention head combination
 - Multiple trees are combined using ASTRAL in order to yield a final consensus tree. 
