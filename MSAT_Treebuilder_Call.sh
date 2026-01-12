#!/bin/bash

# Define directories and paths
ALIGNMENTS_DIR="/home/rohan/PLM-Phylo/Data/Alignments/OMAM20/OMAM20_34sppAligned/"
OUTPUT_DIR="/home/rohan/PLM-Phylo/Results/Trees/MSAT_Trees/OMAM20_34sppAligned/"
ASTRAL_PATH="/home/rohan/Tools/Astral.5.7.8/Astral/astral.5.7.8.jar"

# Loop through all .fas files in the directory
for file in "$ALIGNMENTS_DIR"/*.fas; do
    if [ -f "$file" ]; then
        # Extract the filename without the path
        filename=$(basename "$file")

        # Extract the gene ID (between the two underscores)
        gene_id=$(echo "$filename" | cut -d'_' -f1)

        # Get the length of the alignment (first sequence length)
        seq_length=$(awk '/^>/ {if (seqlen) {print seqlen; exit}} {seqlen += length($0)}' "$file")

        # Skip if the alignment length exceeds 1000 residues
        if [ "$seq_length" -gt 1000 ]; then
            echo "Skipping $filename (Alignment length: $seq_length residues)"
            continue
        fi

        # Run the Python script
        echo "Processing $filename (Alignment length: $seq_length residues)"
        python3 /home/rohan/ESM/Scripts/MSA_Transformer/MSAT_Treebuilder.py \
            --msa "$file" \
            --out "$OUTPUT_DIR" \
            --domain "$gene_id" \
            --astral_path "$ASTRAL_PATH"
    fi
done

echo "Batch processing complete!"
