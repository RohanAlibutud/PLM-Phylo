#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tibble)
  library(ape)
  library(phangorn)
})

# ============================================================
# EDIT ME
# ============================================================

RESULTS_ROOT  <- "/home/rohan/PLM-Phylo/Results/Trees/ESM_35M_Trees/Attention_Trees/OMAM20/OMAM20_4mammals/Trees"
REF_TREE_PATH <- "/home/rohan/PLM-Phylo/Data/Trees/OMAM_supertree_pruned_34spp.nwk"

OUT_DIR <- "/home/rohan/PLM-Phylo/Plots/ESM_35M/OMAM20_4mammals"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

PER_PROTEIN_DIR <- file.path(OUT_DIR, "ESM_layerhead_correctness_per_protein")
dir.create(PER_PROTEIN_DIR, recursive = TRUE, showWarnings = FALSE)

OUT_PREFIX <- "ESM_35M_4mammals"

# Correct if unrooted RF <= threshold
CORRECT_RF_THRESHOLD <- 0

# Optional overrides (set to NA to auto-detect from first protein)
OVERRIDE_LAYERS <- NA_integer_
OVERRIDE_HEADS  <- NA_integer_

# ============================================================
# Helpers
# ============================================================

standardize_tree <- function(tr) {
  tr$edge.length <- NULL
  unroot(tr)
}

read_ref <- function(path) {
  if (!file.exists(path)) stop(paste("Missing reference tree:", path))
  standardize_tree(read.tree(path))
}

rf_unrooted <- function(tr1, tr2) {
  taxa <- intersect(tr1$tip.label, tr2$tip.label)
  if (length(taxa) < 3) return(NA_real_)
  tr1s <- drop.tip(tr1, setdiff(tr1$tip.label, taxa))
  tr2s <- drop.tip(tr2, setdiff(tr2$tip.label, taxa))
  tr1s <- standardize_tree(tr1s)
  tr2s <- standardize_tree(tr2s)
  as.numeric(RF.dist(tr1s, tr2s, normalize = FALSE))
}

parse_head <- function(x) {
  m <- regmatches(x, regexpr("head[0-9]+", x, ignore.case = TRUE))
  if (length(m) == 1 && nchar(m) > 0) return(as.integer(gsub("[^0-9]", "", m)))
  return(NA_integer_)
}

parse_layer_from_dir <- function(layer_dirname) {
  # expects basename like "layer12"
  if (!grepl("^layer[0-9]+$", layer_dirname)) return(NA_integer_)
  as.integer(gsub("^layer", "", layer_dirname))
}

# Find where layer*/ folders actually live for a domain.
# Handles either:
#   domain_dir/layer12
#   domain_dir/domain/layer12
find_inner_dir <- function(domain_dir) {
  dom <- basename(domain_dir)
  cands <- c(file.path(domain_dir, dom), domain_dir)
  cands <- cands[dir.exists(cands)]
  if (length(cands) == 0) return(NA_character_)
  
  has_layers <- function(p) {
    any(grepl("^layer[0-9]+$", basename(list.dirs(p, full.names = TRUE, recursive = FALSE))))
  }
  
  # Prefer the candidate that actually contains layer* folders
  for (p in cands) {
    if (has_layers(p)) return(p)
  }
  # Otherwise fallback to first existing
  cands[1]
}

list_layer_dirs <- function(inner) {
  layer_dirs <- list.dirs(inner, full.names = TRUE, recursive = FALSE)
  layer_dirs[grepl("^layer[0-9]+$", basename(layer_dirs))]
}

list_tree_files <- function(layer_path) {
  # strict: end with _tree.nwk (your files look like *_tree.nwk)
  list.files(layer_path, full.names = TRUE, recursive = FALSE, pattern = "_tree\\.nwk$", ignore.case = TRUE)
}

# ============================================================
# NEW: Detect heads/layers from FIRST protein folder
# ============================================================

detect_grid_from_first_domain <- function(results_root) {
  domain_dirs <- list.dirs(results_root, full.names = TRUE, recursive = FALSE)
  domain_dirs <- domain_dirs[file.info(domain_dirs)$isdir]
  if (length(domain_dirs) == 0) stop(paste("No domain folders under:", results_root))
  
  for (domain_dir in domain_dirs) {
    inner <- find_inner_dir(domain_dir)
    if (is.na(inner)) next
    
    layer_dirs <- list_layer_dirs(inner)
    if (length(layer_dirs) == 0) next
    
    # infer layer nums from folder names
    layer_nums <- sort(unique(vapply(basename(layer_dirs), parse_layer_from_dir, integer(1))))
    
    # infer heads by scanning any tree files we can find
    tree_files_all <- unlist(lapply(layer_dirs, list_tree_files), use.names = FALSE)
    if (length(tree_files_all) == 0) {
      # still return layers; heads unknown
      return(list(domain = basename(domain_dir), inner = inner,
                  layer_levels = layer_nums, head_levels = integer(0)))
    }
    
    heads <- vapply(basename(tree_files_all), parse_head, integer(1))
    heads <- sort(unique(heads[!is.na(heads)]))
    
    return(list(
      domain = basename(domain_dir),
      inner = inner,
      layer_levels = layer_nums,
      head_levels = heads
    ))
  }
  
  stop("Could not find any domain with layer*/ and *_tree.nwk files to infer heads/layers.")
}

# ============================================================
# Main
# ============================================================

ref_tree <- read_ref(REF_TREE_PATH)

# Detect expected grid from first usable protein
grid_info <- detect_grid_from_first_domain(RESULTS_ROOT)

LAYER_LEVELS <- grid_info$layer_levels
HEAD_LEVELS  <- grid_info$head_levels

if (!is.na(OVERRIDE_LAYERS)) {
  # Keep indexing style consistent with what was detected (min layer number)
  if (length(LAYER_LEVELS) > 0 && min(LAYER_LEVELS) == 0L) {
    LAYER_LEVELS <- 0:(OVERRIDE_LAYERS - 1L)
  } else {
    LAYER_LEVELS <- 1:OVERRIDE_LAYERS
  }
}

if (!is.na(OVERRIDE_HEADS)) {
  if (length(HEAD_LEVELS) > 0 && min(HEAD_LEVELS) == 0L) {
    HEAD_LEVELS <- 0:(OVERRIDE_HEADS - 1L)
  } else {
    HEAD_LEVELS <- 1:OVERRIDE_HEADS
  }
}

# If head detection failed (shouldn't for your case), default to 1:20
if (length(HEAD_LEVELS) == 0) HEAD_LEVELS <- 1:20

EXPECTED_LAYERS <- length(LAYER_LEVELS)
EXPECTED_HEADS  <- length(HEAD_LEVELS)
EXPECTED_TILES  <- EXPECTED_LAYERS * EXPECTED_HEADS

cat("\n=== Grid autodetect ===\n")
cat("First domain used for detection:", grid_info$domain, "\n")
cat("Layer levels:", paste(LAYER_LEVELS, collapse = ","), "\n")
cat("Head  levels:", paste(HEAD_LEVELS,  collapse = ","), "\n")
cat("EXPECTED_LAYERS:", EXPECTED_LAYERS, " EXPECTED_HEADS:", EXPECTED_HEADS, " EXPECTED_TILES:", EXPECTED_TILES, "\n\n")

# Scan all domains
domain_dirs <- list.dirs(RESULTS_ROOT, full.names = TRUE, recursive = FALSE)
domain_dirs <- domain_dirs[file.info(domain_dirs)$isdir]
if (length(domain_dirs) == 0) stop(paste("No domain folders under:", RESULTS_ROOT))

rows <- list()

for (domain_dir in domain_dirs) {
  domain <- basename(domain_dir)
  inner <- find_inner_dir(domain_dir)
  if (is.na(inner) || !dir.exists(inner)) next
  
  layer_dirs <- list_layer_dirs(inner)
  if (length(layer_dirs) == 0) next
  
  for (layer_path in layer_dirs) {
    layer_num <- parse_layer_from_dir(basename(layer_path))
    if (is.na(layer_num)) next
    
    tree_files <- sort(list_tree_files(layer_path))
    if (length(tree_files) == 0) next
    
    for (tf in tree_files) {
      head_num <- parse_head(basename(tf))
      if (is.na(head_num)) next
      
      tr <- standardize_tree(read.tree(tf))
      rf <- rf_unrooted(tr, ref_tree)
      correct <- ifelse(is.na(rf), NA, rf <= CORRECT_RF_THRESHOLD)
      
      rows[[length(rows) + 1]] <- tibble(
        domain = domain,
        layer_num = layer_num,
        head = head_num,
        rf = rf,
        correct = correct,
        source = tf
      )
    }
  }
}

df <- bind_rows(rows)
if (nrow(df) == 0) stop("No *_tree.nwk files were scored. Check RESULTS_ROOT structure and filename pattern '*_tree.nwk'.")

# ============================================================
# Count check (per-domain validation against autodetected grid)
# ============================================================

count_check <- df %>%
  group_by(domain) %>%
  summarize(
    layers_found = n_distinct(layer_num),
    heads_found  = n_distinct(head),
    tiles_found  = n(),
    expected_tiles = EXPECTED_TILES,
    missing_tiles  = expected_tiles - tiles_found,
    .groups = "drop"
  ) %>%
  arrange(desc(missing_tiles))

write_csv(df, file.path(OUT_DIR, paste0(OUT_PREFIX, "_layerhead_correctness_results.csv")))
write_csv(count_check, file.path(OUT_DIR, paste0(OUT_PREFIX, "_layerhead_count_check.csv")))
cat("Wrote count check to:\n", file.path(OUT_DIR, paste0(OUT_PREFIX, "_layerhead_count_check.csv")), "\n", sep = "")
print(count_check)

# ============================================================
# Per-protein correctness summary (counts + percent)
# ============================================================

protein_summary <- df %>%
  group_by(domain) %>%
  summarize(
    n_scored    = sum(!is.na(correct)),
    n_correct   = sum(correct == TRUE,  na.rm = TRUE),
    n_incorrect = sum(correct == FALSE, na.rm = TRUE),
    pct_correct = ifelse(n_scored > 0, 100 * n_correct / n_scored, NA_real_),
    .groups = "drop"
  ) %>%
  arrange(desc(pct_correct))

write_csv(protein_summary, file.path(OUT_DIR, paste0(OUT_PREFIX, "_protein_correctness_summary.csv")))
cat("\nPer-protein correctness summary:\n")
print(protein_summary)

# ============================================================
# Plotting (position-mapped so labels ALWAYS line up)
# ============================================================

fill_scale <- scale_fill_manual(
  values = c("TRUE" = "green3", "FALSE" = "red3"),
  na.value = "grey80",
  name = "Correct"
)

# map actual head/layer IDs to sequential positions 1..H / 1..L
HEAD_POS  <- setNames(seq_along(HEAD_LEVELS),  as.character(HEAD_LEVELS))
LAYER_POS <- setNames(seq_along(LAYER_LEVELS), as.character(LAYER_LEVELS))

domains <- sort(unique(df$domain))

for (dom in domains) {
  dsub <- df %>% filter(domain == dom)
  
  grid <- expand.grid(
    layer_num = LAYER_LEVELS,
    head      = HEAD_LEVELS
  ) %>% as_tibble()
  
  dsub_full <- grid %>%
    left_join(dsub, by = c("layer_num","head")) %>%
    mutate(
      domain = dom,
      x = unname(HEAD_POS[as.character(head)]),
      y = unname(LAYER_POS[as.character(layer_num)])
    )
  
  p <- ggplot(dsub_full, aes(x = x, y = y, fill = as.character(correct))) +
    geom_tile(color = "white", linewidth = 0.2) +
    fill_scale +
    scale_x_continuous(
      breaks = seq_along(HEAD_LEVELS),
      labels = HEAD_LEVELS,
      expand = c(0, 0)
    ) +
    scale_y_reverse(
      breaks = seq_along(LAYER_LEVELS),
      labels = LAYER_LEVELS,
      expand = c(0, 0)
    ) +
    coord_fixed() +
    labs(title = dom, x = "Head", y = "Layer") +
    theme_minimal(base_size = 12) +
    theme(panel.grid = element_blank())
  
  ggsave(
    filename = file.path(PER_PROTEIN_DIR, paste0(dom, "_layerhead_correctness.png")),
    plot = p, width = 8, height = 6, dpi = 200
  )
}

# Combined facet plot
df_all <- df %>%
  mutate(
    x = unname(HEAD_POS[as.character(head)]),
    y = unname(LAYER_POS[as.character(layer_num)])
  )

p_all <- ggplot(df_all, aes(x = x, y = y, fill = as.character(correct))) +
  geom_tile(color = "white", linewidth = 0.15) +
  fill_scale +
  scale_x_continuous(
    breaks = seq_along(HEAD_LEVELS),
    labels = HEAD_LEVELS,
    expand = c(0, 0)
  ) +
  scale_y_reverse(
    breaks = seq_along(LAYER_LEVELS),
    labels = LAYER_LEVELS,
    expand = c(0, 0)
  ) +
  coord_fixed() +
  labs(
    title = paste0(OUT_PREFIX, ": Correct (green) vs Incorrect (red)"),
    x = "Head", y = "Layer"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid = element_blank()) +
  facet_wrap(~ domain, ncol = 4)

ggsave(
  filename = file.path(OUT_DIR, paste0(OUT_PREFIX, "_layerhead_correctness.png")),
  plot = p_all, width = 16, height = 12, dpi = 200
)

cat("\nDone.\n")
cat("Per-protein plots written to:\n", PER_PROTEIN_DIR, "\n", sep = "")
cat("Summary CSVs written to:\n", OUT_DIR, "\n", sep = "")
