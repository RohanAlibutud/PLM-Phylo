#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tibble)
  library(ape)
  library(phangorn)
})

# =========================
# EDIT ME
# =========================

RESULTS_ROOT <- "/home/rohan/PLM-Phylo/Results/Trees/ESM_Trees/Attention_Trees/OMAM20/OMAM20_4mammals/Trees"

# Updated per your note
REF_TREE_PATH <- "/home/rohan/PLM-Phylo/Data/Trees/OMAM_supertree_pruned_34spp.nwk"

# Output folder
OUT_DIR <- "/home/rohan/PLM-Phylo/Plots"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Put per-protein plots in their own folder (cleaner)
PER_PROTEIN_DIR <- file.path(OUT_DIR, "ESM_layerhead_correctness_per_protein")
dir.create(PER_PROTEIN_DIR, recursive = TRUE, showWarnings = FALSE)

# Correct if unrooted RF == 0
CORRECT_RF_THRESHOLD <- 0

EXPECTED_LAYERS <- 33
EXPECTED_HEADS  <- 20
EXPECTED_TILES  <- EXPECTED_LAYERS * EXPECTED_HEADS

# =========================
# Helpers
# =========================

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

# =========================
# Main scan
# =========================

ref_tree <- read_ref(REF_TREE_PATH)

domain_dirs <- list.dirs(RESULTS_ROOT, full.names = TRUE, recursive = FALSE)
domain_dirs <- domain_dirs[file.info(domain_dirs)$isdir]
if (length(domain_dirs) == 0) stop(paste("No domain folders under:", RESULTS_ROOT))

rows <- list()

for (domain_dir in domain_dirs) {
  domain <- basename(domain_dir)
  inner <- file.path(domain_dir, domain)  # e.g. TF_4mammals/TF_4mammals
  if (!dir.exists(inner)) next

  layer_dirs <- list.dirs(inner, full.names = TRUE, recursive = FALSE)
  layer_dirs <- layer_dirs[grepl("^layer[0-9]+$", basename(layer_dirs))]

  for (layer_path in layer_dirs) {
    layer_num <- as.integer(gsub("layer", "", basename(layer_path)))

    # Only score Newick head trees
    tree_files <- list.files(layer_path, full.names = TRUE, recursive = FALSE, pattern = "_tree\\.nwk$")
    if (length(tree_files) == 0) next
    tree_files <- sort(tree_files)

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
if (nrow(df) == 0) stop("No head tree .nwk files were scored. Check filename pattern '*_tree.nwk'.")

# =========================
# Count check (explicit 660 validation)
# =========================

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

write_csv(df, file.path(OUT_DIR, "layerhead_correctness_results.csv"))
write_csv(count_check, file.path(OUT_DIR, "layerhead_count_check.csv"))

print(count_check)
# =========================
# Per-protein correctness summary (counts + percent)
# =========================

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

# Write CSV
write_csv(protein_summary, file.path(OUT_DIR, "protein_correctness_summary.csv"))

# Print to console
cat("\nPer-protein correctness summary:\n")
print(protein_summary)

# =========================
# Plotting
# =========================

fill_scale <- scale_fill_manual(
  values = c("TRUE" = "green3", "FALSE" = "red3"),
  na.value = "grey80",
  name = "Correct"
)

domains <- sort(unique(df$domain))

# Save EVERY protein as its own plot file (one per domain)
for (dom in domains) {
  dsub <- df %>% filter(domain == dom)

  # Force full 33x20 grid (missing combos show grey)
  grid <- expand.grid(
    layer_num = 1:EXPECTED_LAYERS,
    head = 1:EXPECTED_HEADS
  ) %>% as_tibble()

  dsub_full <- grid %>%
    left_join(dsub, by = c("layer_num","head")) %>%
    mutate(domain = dom)

  p <- ggplot(dsub_full, aes(x = head, y = layer_num, fill = as.character(correct))) +
    geom_tile(color = "white", linewidth = 0.2) +
    fill_scale +
    scale_x_continuous(breaks = 1:20) +
    scale_y_reverse(breaks = c(1,5,10,15,20,25,30,33)) +
    labs(title = dom, x = "Head", y = "Layer") +
    theme_minimal(base_size = 12) +
    theme(panel.grid = element_blank())

  # Saved per-protein file:
  ggsave(
    filename = file.path(PER_PROTEIN_DIR, paste0(dom, "_layerhead_correctness.png")),
    plot = p, width = 8, height = 6, dpi = 200
  )
}

# Combined facet plot (kept, saved to OUT_DIR)
p_all <- ggplot(df, aes(x = head, y = layer_num, fill = as.character(correct))) +
  geom_tile(color = "white", linewidth = 0.15) +
  fill_scale +
  scale_x_continuous(breaks = c(1,5,10,15,20)) +
  scale_y_reverse(breaks = c(1,5,10,15,20,25,30,33)) +
  labs(title = "ESM Attention Trees: Correct (green) vs Incorrect (red)", x = "Head", y = "Layer") +
  theme_minimal(base_size = 11) +
  theme(panel.grid = element_blank()) +
  facet_wrap(~ domain, ncol = 4)

ggsave(
  filename = file.path(OUT_DIR, "ALL_domains_layerhead_correctness.png"),
  plot = p_all, width = 16, height = 12, dpi = 200
)

cat("\nDone.\n")
cat("Per-protein plots written to:\n", PER_PROTEIN_DIR, "\n", sep = "")
cat("Summary CSVs written to:\n", OUT_DIR, "\n", sep = "")
