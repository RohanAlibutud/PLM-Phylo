#!/usr/bin/env bash
set -euo pipefail

# Project root (absolute)
PROJECT_ROOT="/home/rohan/PLM-Phylo"

# Input folders (absolute)
IN_UNALIGNED="${PROJECT_ROOT}/Data/Alignments/OMAM20/OMAM20_34spp"
IN_ALIGNED="${PROJECT_ROOT}/Data/Alignments/OMAM20/OMAM20_34sppAligned"

# Output folders (absolute)
OUT_UNALIGNED="${PROJECT_ROOT}/Data/Alignments/OMAM20/OMAM20_4mammals"
OUT_ALIGNED="${PROJECT_ROOT}/Data/Alignments/OMAM20/OMAM20_4mammalsAligned"

mkdir -p "$OUT_UNALIGNED" "$OUT_ALIGNED"

# The 4 mammals to keep (exact ID expected as first token after '>')
KEEP_SET=(
  "Homo_sapiens"
  "Macaca_fascicularis"
  "Rattus_norvegicus"
  "Monodelphis_domestica"
)

# Build an awk regex like: ^(Homo_sapiens|Macaca_fascicularis|...)$
KEEP_REGEX="$(printf "%s|" "${KEEP_SET[@]}")"
KEEP_REGEX="${KEEP_REGEX%|}"

filter_fasta_4mammals() {
  local in_file="$1"
  local out_file="$2"

  # Single-line FASTA assumed: header line then exactly one sequence line.
  # We match the ID as the first token after '>' up to whitespace.
  awk -v re="^(${KEEP_REGEX})$" '
    BEGIN { keep=0 }
    /^>/ {
      id=$0
      sub(/^>/,"",id)
      sub(/[ \t\r].*$/,"",id)
      keep = (id ~ re)
      if (keep) print $0
      next
    }
    {
      if (keep) print $0
      keep=0
    }
  ' "$in_file" > "$out_file"
}

echo "Filtering unaligned: $IN_UNALIGNED -> $OUT_UNALIGNED"
shopt -s nullglob
for f in "${IN_UNALIGNED}"/*.fas; do
  bn="$(basename "$f")"
  out="${OUT_UNALIGNED}/${bn/_34spp/_4mammals}"
  filter_fasta_4mammals "$f" "$out"
done

echo "Filtering aligned: $IN_ALIGNED -> $OUT_ALIGNED"
for f in "${IN_ALIGNED}"/*.fas; do
  bn="$(basename "$f")"
  out="${OUT_ALIGNED}/${bn/_34sppAligned/_4mammalsAligned}"
  filter_fasta_4mammals "$f" "$out"
done

echo "Done."
echo "Created:"
echo "  $OUT_UNALIGNED"
echo "  $OUT_ALIGNED"
