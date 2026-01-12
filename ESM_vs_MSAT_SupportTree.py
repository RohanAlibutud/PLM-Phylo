#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import defaultdict
from ete3 import Tree, TreeStyle, NodeStyle, TextFace

# === Parameters ===
protein_set = "34spp"

esm_results_folder = (
    f"/home/rohan/PLM-Phylo/Results/Trees/ESM_35M_Trees/Attention_Trees/OMAM20/OMAM20_{protein_set}/Trees"
)

# FIXED: match your actual MSAT layout shown in terminal
msat_results_folder = (
    f"/home/rohan/PLM-Phylo/Results/Trees/MSAT_Trees/OMAM20_{protein_set}Aligned"
)

pruned_tree_file = "/home/rohan/ESM/Data/Trees/OMAM_supertree_pruned_34spp.nwk"

# Where to write figures
fig_outdir = f"/home/rohan/PLM-Phylo/Results/Figures/OMAM20_{protein_set}/Trees"
os.makedirs(fig_outdir, exist_ok=True)

# === Load supertree ===
supertree = Tree(pruned_tree_file, format=1)
full_set = set(leaf.name for leaf in supertree.iter_leaves())

# === Define function to get bipartitions ===
def get_bipartitions(tree: Tree):
    biparts = []
    tips = set(leaf.name for leaf in tree.iter_leaves())
    for node in tree.traverse():
        if node.is_leaf():
            continue
        split = set(leaf.name for leaf in node.iter_leaves())
        # canonicalize so we always store the "smaller side" of the split
        if len(split) > len(tips) / 2:
            split = tips - split
        biparts.append(frozenset(split))
    return biparts

reference_biparts = get_bipartitions(supertree)

# === Score support ===
esm_counts = defaultdict(int)
msat_counts = defaultdict(int)

# -------------------------
# --- ESM Trees (unchanged)
# -------------------------
esm_files = []
for root, dirs, files in os.walk(esm_results_folder):
    # keep your existing skip logic
    if any(skip in root for skip in ["Consensus", "Replicates"]):
        continue
    for fn in files:
        if fn.endswith(".nwk"):
            esm_files.append(os.path.join(root, fn))

print(f"Found {len(esm_files)} ESM trees.")

for fp in esm_files:
    try:
        t = Tree(fp, format=1)
        biparts = set(get_bipartitions(t))
        for bip in reference_biparts:
            if bip in biparts:
                esm_counts[bip] += 1
    except Exception as e:
        print(f"[Warning] Skipping ESM tree {fp}: {e}")

# -----------------------------------------
# --- MSAT Trees (FIXED: correct structure)
# -----------------------------------------
msat_files = []

# We want the per-layerhead NJ trees under each domain:
# {msat_results_folder}/{DOMAIN}/Trees/aDist_Trees/*.nwk
for domain in os.listdir(msat_results_folder):
    dom_dir = os.path.join(msat_results_folder, domain)
    if not os.path.isdir(dom_dir):
        continue

    adist_dir = os.path.join(dom_dir, "Trees", "aDist_Trees")
    if not os.path.isdir(adist_dir):
        # (some domains might be incomplete; just skip)
        continue

    for fn in os.listdir(adist_dir):
        # your MSAT layerhead trees look like: DOMAIN_layer10_head11_aDist_tree.nwk
        if fn.endswith("_aDist_tree.nwk") and fn.endswith(".nwk"):
            msat_files.append(os.path.join(adist_dir, fn))

print(f"Found {len(msat_files)} MSAT aDist layerhead trees.")

for fp in msat_files:
    try:
        t = Tree(fp, format=1)
        biparts = set(get_bipartitions(t))
        for bip in reference_biparts:
            if bip in biparts:
                msat_counts[bip] += 1
    except Exception as e:
        print(f"[Warning] Skipping MSAT tree {fp}: {e}")

# === Normalize ===
n_esm = len(esm_files)
n_msat = len(msat_files)

def safe_pct(count: int, denom: int) -> float:
    return (count / denom) * 100.0 if denom > 0 else 0.0

esm_support = {bip: safe_pct(esm_counts[bip], n_esm) for bip in reference_biparts}
msat_support = {bip: safe_pct(msat_counts[bip], n_msat) for bip in reference_biparts}

# === Annotate and visualize ===
def annotate_and_export(tree, esm_support, msat_support):
    for node in tree.traverse():
        if node.is_leaf():
            continue

        leaves = set(leaf.name for leaf in node.iter_leaves())
        if len(leaves) > len(full_set) / 2:
            leaves = full_set - leaves
        bip = frozenset(leaves)

        esm = esm_support.get(bip, None)
        msat = msat_support.get(bip, None)

        if esm is not None and n_esm > 0:
            esm_face = TextFace(f"{esm:.1f}%", fsize=6, fgcolor="dodgerblue")
            node.add_face(esm_face, column=0, position="branch-top")

        if msat is not None and n_msat > 0:
            msat_face = TextFace(f"{msat:.1f}%", fsize=6, fgcolor="orange")
            node.add_face(msat_face, column=0, position="branch-bottom")

        nstyle = NodeStyle()
        nstyle["size"] = 0
        node.set_style(nstyle)

    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.show_branch_support = False
    ts.scale = 140

    ts.legend.add_face(TextFace("esm-pFM RAM", fsize=8, fgcolor="dodgerblue"), column=0)
    ts.legend.add_face(TextFace("msat-pFM SAM",   fsize=8, fgcolor="orange"), column=0)

    svg_path = os.path.join(fig_outdir, "annotated_tree.svg")
    png_path = os.path.join(fig_outdir, "annotated_tree.png")

    tree.render(svg_path, tree_style=ts)
    tree.render(png_path, tree_style=ts, w=800, units="px")
    tree.show()

    print(f"Exported:\n  {svg_path}\n  {png_path}")

annotate_and_export(supertree, esm_support, msat_support)
