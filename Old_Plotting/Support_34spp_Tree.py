import os
import glob
from collections import defaultdict
from ete3 import Tree, TreeStyle, NodeStyle, TextFace

# === Parameters ===
protein_set = "34spp"
esm_results_folder = f"/home/rohan/ESM/Results/OMAM20/OMAM20_{protein_set}/Trees"
ml_results_folder = f"/home/rohan/ESM/Results/OMAM20/ML_MFP/OMAM20_{protein_set}Aligned"
pruned_tree_file = "/home/rohan/ESM/Data/Trees/OMAM_supertree_pruned_34spp.nwk"

# === Load supertree ===
supertree = Tree(pruned_tree_file, format=1)
full_set = set(leaf.name for leaf in supertree.iter_leaves())

# === Define function to get bipartitions ===
def get_bipartitions(tree):
    biparts = []
    tips = set(leaf.name for leaf in tree.iter_leaves())
    for node in tree.traverse():
        if not node.is_leaf():
            split = set(leaf.name for leaf in node.iter_leaves())
            if len(split) > len(tips) / 2:
                split = tips - split
            biparts.append(frozenset(split))
    return biparts

reference_biparts = get_bipartitions(supertree)

# === Score support ===
esm_counts = defaultdict(int)
ml_counts = defaultdict(int)

# --- ESM Trees ---
esm_files = []
for root, dirs, files in os.walk(esm_results_folder):
    if any(skip in root for skip in ["Consensus", "Replicates"]):
        continue
    for file in files:
        if file.endswith(".nwk"):
            esm_files.append(os.path.join(root, file))

print(f"Found {len(esm_files)} ESM trees.")

for file in esm_files:
    try:
        trees = Tree(file, format=1)
        biparts = set(get_bipartitions(trees))
        for bip in reference_biparts:
            if bip in biparts:
                esm_counts[bip] += 1
    except Exception as e:
        print(f"[Warning] Skipping {file}: {e}")

# --- ML Trees ---
ml_files = glob.glob(os.path.join(ml_results_folder, "*.treefile"))
print(f"Found {len(ml_files)} ML trees.")

for file in ml_files:
    try:
        tree = Tree(file, format=1)
        biparts = set(get_bipartitions(tree))
        for bip in reference_biparts:
            if bip in biparts:
                ml_counts[bip] += 1
    except Exception as e:
        print(f"[Warning] Skipping {file}: {e}")

# === Normalize ===
n_esm = len(esm_files)
n_ml = len(ml_files)
esm_support = {bip: (esm_counts[bip] / n_esm) * 100 for bip in reference_biparts}
ml_support = {bip: (ml_counts[bip] / n_ml) * 100 for bip in reference_biparts}

# === Annotate and visualize ===
def annotate_and_show(tree, esm_support, ml_support):
    for node in tree.traverse():
        if not node.is_leaf():
            leaves = set(leaf.name for leaf in node.iter_leaves())
            if len(leaves) > len(full_set) / 2:
                leaves = full_set - leaves
            bip = frozenset(leaves)

            esm = esm_support.get(bip, None)
            ml = ml_support.get(bip, None)

            if esm is not None:
                esm_face = TextFace(f"{esm:.1f}%", fsize=6, fgcolor="dodgerblue")
                node.add_face(esm_face, column=0, position="branch-top")

            if ml is not None:
                ml_face = TextFace(f"{ml:.1f}%", fsize=6, fgcolor="darkgreen")
                node.add_face(ml_face, column=0, position="branch-bottom")

            nstyle = NodeStyle()
            nstyle["size"] = 0
            node.set_style(nstyle)

    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.show_branch_support = False
    ts.scale = 140
    #ts.title.add_face(TextFace("dodgerblue = PLM-ss | darkgreen = ML+MFP", fsize=8), column=0)
    # ts.tip_font_size is not supported in TreeStyle; shrink tips manually below

    # Add legend using TextFaces manually positioned with add_face
    legend_face1 = TextFace("PLM-ss", fsize=8, fgcolor="dodgerblue")
    legend_face2 = TextFace("ML+MFP", fsize=8, fgcolor="darkgreen")
    ts.legend.add_face(legend_face1, column=0)
    ts.legend.add_face(legend_face2, column=0)



    tree.show(tree_style=ts)
    tree.render(f"/home/rohan/ESM/Results/OMAM20/OMAM20_{protein_set}/Trees/annotated_tree.svg", tree_style=ts)
    tree.render(f"/home/rohan/ESM/Results/OMAM20/OMAM20_{protein_set}/Trees/annotated_tree.png", tree_style=ts, w=800, units="px")
    print("Tree exported to annotated_tree.svg and annotated_tree.png")

# Annotate and export tree
annotate_and_show(supertree, esm_support, ml_support)
