 #!/usr/bin/env bash
set -euo pipefail

# ==== Paths (absolute) ====
INPUT_DIR="/home/rohan/PLM-Phylo/Data/Alignments/OMAM20/OMAM20_4mammals"
OUT_DIR="/home/rohan/PLM-Phylo/Results/Trees/ESM_35M_Trees/Attention_Trees/OMAM20/OMAM20_4mammals"
SCRIPT="/home/rohan/ESM/Scripts/ESM_Treebuilder/ESM_Treebuild_PSA.py"
MODEL="/home/rohan/ESM/Models/ESM2_35M_FULLMODULE.pt"
TOKENIZER="/home/rohan/ESM/Models/ESM2_alphabet.pkl"
ASTRAL="/home/rohan/Tools/Astral.5.7.8/Astral/astral.5.7.8.jar"

DISTANCE="psa_gapless"
PW_ALIGN="mismatch"

mkdir -p "$OUT_DIR"

# Collect candidate FASTA-like files (adjust patterns if needed)
mapfile -t FASTA_FILES < <(
  find "$INPUT_DIR" -maxdepth 1 -type f \( \
      -iname "*.fas" -o -iname "*.fasta" -o -iname "*.faa" -o -iname "*.fa" \
    \) | sort
)

if [[ ${#FASTA_FILES[@]} -eq 0 ]]; then
  echo "ERROR: No FASTA files found in: $INPUT_DIR" >&2
  exit 1
fi

echo "Found ${#FASTA_FILES[@]} input files in: $INPUT_DIR"
echo "Output root: $OUT_DIR"
echo

# Optional: simple run log
LOGFILE="$OUT_DIR/run_$(date +%Y%m%d_%H%M%S).log"
touch "$LOGFILE"

for INPUT in "${FASTA_FILES[@]}"; do
  fname="$(basename "$INPUT")"
  domain="${fname%.*}"   # strip extension

  echo "=== Running domain: $domain ===" | tee -a "$LOGFILE"
  echo "Input: $INPUT" | tee -a "$LOGFILE"

  python3 "$SCRIPT" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --input "$INPUT" \
    --out "$OUT_DIR" \
    --domain "$domain" \
    --distance "$DISTANCE" \
    --pw_align "$PW_ALIGN" \
    --astral "$ASTRAL" 2>&1 | tee -a "$LOGFILE"

  echo | tee -a "$LOGFILE"
done

echo "All runs complete."
echo "Log: $LOGFILE"
4apes