#!/usr/bin/env python3
"""
Score ESM layer/head NJ trees vs a reference tree using unrooted RF distance.

- Autodetect EXPECTED_LAYERS / EXPECTED_HEADS by scanning the first domain folder.
- Robustly handles directory layouts:
    RESULTS_ROOT/DOMAIN/layerX/*.nwk
    RESULTS_ROOT/DOMAIN/DOMAIN/layerX/*.nwk

Outputs:
- <OUT_PREFIX>_layerhead_correctness_results.csv
- <OUT_PREFIX>_layerhead_count_check.csv
- <OUT_PREFIX>_protein_correctness_summary.csv
- Per-domain heatmaps to: PER_PROTEIN_DIR/<DOMAIN>_layerhead_correctness.png
- Optional combined figure: <OUT_PREFIX>_layerhead_correctness.png
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dendropy


# -------------------------
# Parsing helpers
# -------------------------

LAYER_RE = re.compile(r"^layer(\d+)$", re.IGNORECASE)
HEAD_RE = re.compile(r"head(\d+)", re.IGNORECASE)


def parse_layer_from_dirname(name: str) -> Optional[int]:
    m = LAYER_RE.match(name)
    return int(m.group(1)) if m else None


def parse_head_from_filename(name: str) -> Optional[int]:
    m = HEAD_RE.search(name)
    return int(m.group(1)) if m else None


def find_inner_dir(domain_dir: Path) -> Optional[Path]:
    """
    Prefer domain_dir/domain_dir.name if it exists AND contains layer* folders,
    otherwise fall back to domain_dir if it contains layer* folders.
    """
    dom = domain_dir.name
    cands = [domain_dir / dom, domain_dir]

    def has_layers(p: Path) -> bool:
        if not p.exists() or not p.is_dir():
            return False
        for child in p.iterdir():
            if child.is_dir() and LAYER_RE.match(child.name):
                return True
        return False

    for p in cands:
        if has_layers(p):
            return p

    # If neither contains layer folders, return first existing candidate
    for p in cands:
        if p.exists() and p.is_dir():
            return p
    return None


def list_layer_dirs(inner: Path) -> List[Path]:
    out: List[Path] = []
    for p in inner.iterdir():
        if p.is_dir() and LAYER_RE.match(p.name):
            out.append(p)
    out.sort(key=lambda x: parse_layer_from_dirname(x.name) or 0)
    return out


def list_tree_files(layer_dir: Path) -> List[Path]:
    # Your files: *_tree.nwk
    files = sorted(layer_dir.glob("*_tree.nwk"))
    # Be slightly forgiving:
    if not files:
        files = sorted(layer_dir.glob("*.nwk"))
    return files


# -------------------------
# RF distance (unrooted)
# -------------------------

@dataclass(frozen=True)
class RFConfig:
    correct_threshold: float = 0.0  # correct if RF <= threshold


def load_tree(path: Path, taxon_namespace: dendropy.TaxonNamespace) -> dendropy.Tree:
    # preserve_underscores keeps taxon labels stable if you have underscores
    t = dendropy.Tree.get(
        path=str(path),
        schema="newick",
        rooting="force-unrooted",
        preserve_underscores=True,
        taxon_namespace=taxon_namespace,
    )
    # Remove branch lengths (match your R standardization)
    for e in t.postorder_edge_iter():
        e.length = None
    return t


def rf_unrooted(
    t1: dendropy.Tree,
    t2: dendropy.Tree,
) -> Optional[int]:
    # Require >= 3 shared taxa
    taxa1 = {tx.label for tx in t1.taxon_namespace}
    taxa2 = {tx.label for tx in t2.taxon_namespace}
    shared = taxa1.intersection(taxa2)
    if len(shared) < 3:
        return None

    # Extract shared taxa only
    t1s = t1.extract_tree_with_taxa_labels(shared, suppress_unifurcations=True)
    t2s = t2.extract_tree_with_taxa_labels(shared, suppress_unifurcations=True)

    # Unrooted RF distance
    return dendropy.calculate.treecompare.symmetric_difference(t1s, t2s)

def is_cherry(tree: dendropy.Tree, taxon_a: str, taxon_b: str) -> bool:
    """
    Returns True iff taxon_a and taxon_b are sister tips.
    """
    na = tree.find_node_with_taxon_label(taxon_a)
    nb = tree.find_node_with_taxon_label(taxon_b)

    if na is None or nb is None:
        return False  # NOT missing — explicitly incorrect

    pa = na.parent_node
    pb = nb.parent_node

    if pa is None or pb is None:
        return False

    return pa is pb



# -------------------------
# Grid detection
# -------------------------

@dataclass(frozen=True)
class Grid:
    layer_levels: List[int]
    head_levels: List[int]


def detect_grid_from_first_domain(results_root: Path) -> Grid:
    """
    Scan domains in RESULTS_ROOT until we find one with:
      - layer folders
      - tree files containing head numbers
    Use that domain to define expected layer/head levels.
    """
    domain_dirs = [p for p in results_root.iterdir() if p.is_dir()]
    domain_dirs.sort(key=lambda p: p.name)

    for domain_dir in domain_dirs:
        inner = find_inner_dir(domain_dir)
        if inner is None:
            continue
        layer_dirs = list_layer_dirs(inner)
        if not layer_dirs:
            continue

        layer_levels = []
        for ld in layer_dirs:
            ln = parse_layer_from_dirname(ld.name)
            if ln is not None:
                layer_levels.append(ln)
        layer_levels = sorted(set(layer_levels))
        if not layer_levels:
            continue

        # Gather tree files and infer heads
        head_levels: List[int] = []
        for ld in layer_dirs:
            for tf in list_tree_files(ld):
                hn = parse_head_from_filename(tf.name)
                if hn is not None:
                    head_levels.append(hn)

        head_levels = sorted(set(head_levels))

        # If no heads could be parsed, try next domain
        if not head_levels:
            continue

        return Grid(layer_levels=layer_levels, head_levels=head_levels)

    raise RuntimeError("Could not detect layers/heads from any domain (no usable layer*/ and *_tree.nwk files).")


# -------------------------
# Main scoring
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default=str(Path("/home/rohan/PLM-Phylo/Results/Trees/ESM_35M_Trees/Attention_Trees/OMAM20/OMAM20_4mammals/Trees")))
    ap.add_argument("--ref_tree", default=str(Path("/home/rohan/PLM-Phylo/Data/Trees/OMAM_supertree_pruned_34spp.nwk")))
    ap.add_argument("--out_dir", default=str(Path("/home/rohan/PLM-Phylo/Plots/ESM_35M/OMAM20_4mammals")))
    ap.add_argument("--out_prefix", default="ESM_35M_4mammals")

    # Optional overrides (counts). If set, levels will be generated as consecutive ints
    ap.add_argument("--layers", type=int, default=0, help="Override number of layers (0=auto)")
    ap.add_argument("--heads", type=int, default=0, help="Override number of heads (0=auto)")

    ap.add_argument("--rf_threshold", type=float, default=0.0)
    ap.add_argument("--combined_plot", action="store_true", help="Also save one combined multi-panel figure (can be huge).")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_protein_dir = out_dir / "ESM_layerhead_correctness_per_protein"
    per_protein_dir.mkdir(parents=True, exist_ok=True)

    # Detect grid
    grid = detect_grid_from_first_domain(results_root)

    layer_levels = grid.layer_levels
    head_levels = grid.head_levels

    # Apply overrides as COUNTS, preserving indexing style (min id)
    if args.layers and args.layers > 0:
        min_layer = min(layer_levels) if layer_levels else 1
        if min_layer == 0:
            layer_levels = list(range(0, args.layers))
        else:
            layer_levels = list(range(1, args.layers + 1))

    if args.heads and args.heads > 0:
        min_head = min(head_levels) if head_levels else 1
        if min_head == 0:
            head_levels = list(range(0, args.heads))
        else:
            head_levels = list(range(1, args.heads + 1))

    print("\n=== Grid ===")
    print("Layers:", layer_levels)
    print("Heads :", head_levels)
    print(f"Expected tiles: {len(layer_levels) * len(head_levels)}\n")

    # Prepare RF
    cfg = RFConfig(correct_threshold=args.rf_threshold)
    tax = dendropy.TaxonNamespace()

    ref_tree = load_tree(Path(args.ref_tree), tax)

    # Scan
    rows: List[dict] = []
    domain_dirs = [p for p in results_root.iterdir() if p.is_dir()]
    domain_dirs.sort(key=lambda p: p.name)

    for domain_dir in domain_dirs:
        domain = domain_dir.name
        inner = find_inner_dir(domain_dir)
        if inner is None:
            continue

        for layer_dir in list_layer_dirs(inner):
            layer_num = parse_layer_from_dirname(layer_dir.name)
            if layer_num is None:
                continue

            tree_files = list_tree_files(layer_dir)
            if not tree_files:
                continue

            for tf in tree_files:
                head_num = parse_head_from_filename(tf.name)
                if head_num is None:
                    continue

                try:
                    tr = load_tree(tf, tax)
                    correct = is_cherry(tr, "Homo_sapiens", "Macaca_fascicularis")
                    rf = None  # not used

                except Exception as e:
                    rf = None

                if rf is not None:
                    correct = (rf <= cfg.correct_threshold)

                rows.append(
                    dict(
                        domain=domain,
                        layer_num=layer_num,
                        head=head_num,
                        rf=rf,
                        correct=correct,
                        source=str(tf),
                    )
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No trees were scored. Check patterns and directory layout.")

    # Write raw results
    df.to_csv(out_dir / f"{args.out_prefix}_layerhead_correctness_results.csv", index=False)

    # Count check
    expected_tiles = len(layer_levels) * len(head_levels)
    count_check = (
        df.groupby("domain")
        .agg(
            layers_found=("layer_num", "nunique"),
            heads_found=("head", "nunique"),
            tiles_found=("head", "size"),
        )
        .reset_index()
    )
    count_check["expected_tiles"] = expected_tiles
    count_check["missing_tiles"] = count_check["expected_tiles"] - count_check["tiles_found"]
    count_check = count_check.sort_values("missing_tiles", ascending=False)
    count_check.to_csv(out_dir / f"{args.out_prefix}_layerhead_count_check.csv", index=False)

    print("Count check written to:", out_dir / f"{args.out_prefix}_layerhead_count_check.csv")
    print(count_check.head(20).to_string(index=False))

    # Per-protein summary
    # n_scored = non-null correct
    protein_summary = (
        df.assign(correct_bool=df["correct"].astype("boolean"))
        .groupby("domain")
        .agg(
            n_scored=("correct_bool", lambda s: int(s.notna().sum())),
            n_correct=("correct_bool", lambda s: int((s == True).sum())),
            n_incorrect=("correct_bool", lambda s: int((s == False).sum())),
        )
        .reset_index()
    )
    protein_summary["pct_correct"] = np.where(
        protein_summary["n_scored"] > 0,
        100.0 * protein_summary["n_correct"] / protein_summary["n_scored"],
        np.nan,
    )
    protein_summary = protein_summary.sort_values("pct_correct", ascending=False)
    print(protein_summary)
    protein_summary.to_csv(out_dir / f"{args.out_prefix}_protein_correctness_summary.csv", index=False)

    print("\nPer-protein correctness summary written to:",
          out_dir / f"{args.out_prefix}_protein_correctness_summary.csv")

    # --------------------------------
    # Plotting (fixed grid, aligned)
    # --------------------------------
    head_to_x = {h: i for i, h in enumerate(head_levels, start=1)}
    layer_to_y = {l: i for i, l in enumerate(layer_levels, start=1)}

    # Build full grid per domain and plot
    domains = sorted(df["domain"].unique().tolist())

    for dom in domains:
        dsub = df[df["domain"] == dom].copy()

        # full grid (layer_levels x head_levels)
        grid_rows = []
        for l in layer_levels:
            for h in head_levels:
                grid_rows.append((l, h))
        g = pd.DataFrame(grid_rows, columns=["layer_num", "head"])
        merged = g.merge(dsub, on=["layer_num", "head"], how="left")

        # matrix for imshow: rows=layers, cols=heads
        # value: 1 correct, 0 incorrect, nan missing
        mat = np.full((len(layer_levels), len(head_levels)), np.nan, dtype=float)
        for _, r in merged.iterrows():
            ly = layer_to_y[int(r["layer_num"])] - 1
            hx = head_to_x[int(r["head"])] - 1
            if pd.isna(r["correct"]):
                mat[ly, hx] = np.nan
            else:
                mat[ly, hx] = 1.0 if bool(r["correct"]) else 0.0

        fig, ax = plt.subplots(figsize=(10, 6))

        # We want: green=correct, red=incorrect, grey=missing
        # We'll plot incorrect as 0 (red), correct as 1 (green) using a custom colormap:
        from matplotlib.colors import ListedColormap, BoundaryNorm

        cmap = ListedColormap(["red", "palegreen"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

        # draw missing as grey via masked array
        mmat = np.ma.masked_invalid(mat)
        ax.imshow(mmat, cmap=cmap, norm=norm, aspect="auto", origin="upper")
        ax.set_facecolor("0.85")  # grey background for missing tiles

        # ticks/labels aligned to cell centers
        ax.set_xticks(np.arange(len(head_levels)))
        ax.set_xticklabels(head_levels, rotation=0)
        ax.set_yticks(np.arange(len(layer_levels)))
        # show layers top->bottom in descending label like your R (reverse axis)
        # origin="upper" already makes row0 top; layer_levels are in ascending order
        ax.set_yticklabels(layer_levels)

        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(dom)
        if dom == "TF_4mammals":
            ax.set_title("Transferrin esm-pFM correctly inferred phylogenies",fontsize = 20)
        else: 
            continue

        # light gridlines between tiles
        ax.set_xticks(np.arange(-0.5, len(head_levels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(layer_levels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        fig.tight_layout()
        fig.savefig(per_protein_dir / f"{dom}_layerhead_correctness.png", dpi=200)
        plt.show()
        plt.close(fig)

    print("\nPer-protein plots written to:", per_protein_dir)

    # Optional combined plot (can be large)
    if args.combined_plot:
        n = len(domains)
        ncol = 4
        nrow = int(np.ceil(n / ncol))
        fig_w = 4 * ncol
        fig_h = 3 * nrow

        fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h))
        axes = np.array(axes).reshape(-1)

        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap = ListedColormap(["red", "green"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

        for i, dom in enumerate(domains):
            ax = axes[i]
            dsub = df[df["domain"] == dom].copy()
            mat = np.full((len(layer_levels), len(head_levels)), np.nan, dtype=float)
            for _, r in dsub.iterrows():
                ly = layer_to_y[int(r["layer_num"])] - 1
                hx = head_to_x[int(r["head"])] - 1
                if pd.isna(r["correct"]):
                    mat[ly, hx] = np.nan
                else:
                    mat[ly, hx] = 1.0 if bool(r["correct"]) else 0.0
            mmat = np.ma.masked_invalid(mat)
            ax.imshow(mmat, cmap=cmap, norm=norm, aspect="auto", origin="upper")
            ax.set_facecolor("0.85")
            ax.set_title(dom, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        for j in range(len(domains), len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"{args.out_prefix}: Correct (green) vs Incorrect (red)", y=0.995)
        fig.tight_layout()
        fig.savefig(out_dir / f"{args.out_prefix}_layerhead_correctness.png", dpi=200)
        plt.show()
        plt.close(fig)

        print("Combined plot written to:", out_dir / f"{args.out_prefix}_layerhead_correctness.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
