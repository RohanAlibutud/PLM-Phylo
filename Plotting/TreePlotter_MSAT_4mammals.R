# Install and load required packages
if (!requireNamespace("TreeDist", quietly = TRUE)) install.packages("TreeDist")
if (!requireNamespace("ape", quietly = TRUE)) install.packages("ape")
library(TreeDist)
library(ape)

# File paths
supertree_path <- "/home/rohan/ESM/Data/Trees/OMAM_supertree.nwk"
proteins <- c("AIMP2", "APOBEC2", "AR", "CASQ1", 
              "COCH", "DYRK1B", "GPX3", "HAUS1", 
              "IL6", "INO80B", "KRT23", "NEUROG1",
              "OPN1SW", "PPOX", "RHO", "SLC39A12", 
              "TF", "TPPP2", "UBA2", "UPRT")

# Initialize output dataframe
results <- data.frame(
  Protein = character(),
  Normalized_CI_Distance_ESM_PSA = numeric(),
  Normalized_CI_Distance_MSAT_Treebuilder = numeric(),
  Normalized_CI_Distance_JTT_ML = numeric(),
  stringsAsFactors = FALSE
)

# Load the supertree once
supertree <- read.tree(supertree_path)

# Loop through each protein
for (protein in proteins) {
  tree1_path <- paste0("/home/rohan/ESM/Results/OMAM20/OMAM20_34spp/Trees/", protein, "/Consensus_Trees/", protein, "_astral_consensus_tree.nwk")
  tree2_path <- paste0("/home/rohan/ESM/Results/OMAM20/MSAT/OMAM20_34sppAligned/Attention_Matrices/", protein, "/", protein, "_astral_output.nwk")
  tree3_path <- paste0("/home/rohan/ESM/Results/OMAM20/ML_MFP/OMAM20_34sppAligned/", protein, "_34sppAligned.fas.treefile")
  
  # Try-catch block to handle errors
  tryCatch({
    # Read the trees
    tree1 <- read.tree(tree1_path)
    tree2 <- read.tree(tree2_path)
    tree3 <- read.tree(tree3_path)
    
    # Identify common taxa
    common_taxa <- Reduce(intersect, list(supertree$tip.label, tree1$tip.label, tree2$tip.label, tree3$tip.label))
    
    # Prune trees to retain only common taxa
    supertree_pruned <- drop.tip(supertree, setdiff(supertree$tip.label, common_taxa))
    tree1_pruned <- drop.tip(tree1, setdiff(tree1$tip.label, common_taxa))
    tree2_pruned <- drop.tip(tree2, setdiff(tree2$tip.label, common_taxa))
    tree3_pruned <- drop.tip(tree3, setdiff(tree3$tip.label, common_taxa))
    
    # CI Distance Calculations
    ci_distance_tree1 <- ClusteringInfoDistance(supertree_pruned, tree1_pruned)
    ci_distance_tree2 <- ClusteringInfoDistance(supertree_pruned, tree2_pruned)
    ci_distance_tree3 <- ClusteringInfoDistance(supertree_pruned, tree3_pruned)
    
    # Normalized CI Distance
    normalized_ci_tree1 <- ci_distance_tree1 / log2(length(common_taxa))
    normalized_ci_tree1 <- round(normalized_ci_tree1, digits = 3)
    normalized_ci_tree2 <- ci_distance_tree2 / log2(length(common_taxa))
    normalized_ci_tree2 <- round(normalized_ci_tree2, digits = 3)
    normalized_ci_tree3 <- ci_distance_tree3 / log2(length(common_taxa))
    normalized_ci_tree3 <- round(normalized_ci_tree3, digits = 3)
    
    # Add results to dataframe
    results <- rbind(
      results,
      data.frame(
        Protein = protein,
        Normalized_CI_Distance_ESM = normalized_ci_tree1,
        Normalized_CI_Distance_MSAT = normalized_ci_tree2,
        Normalized_CI_Distance_ML_MFP = normalized_ci_tree3,
        stringsAsFactors = FALSE
      )
    )
    
    cat("Processed:", protein, "\n")
    
  }, error = function(e) {
    cat("Error processing", protein, ":", e$message, "\n")
  })
}

# Write results to CSV
output_csv <- "/home/rohan/ESM/Results/OMAM20_CI.csv"
write.csv(results, output_csv, row.names = FALSE)

cat("Results saved to", output_csv, "\n")
