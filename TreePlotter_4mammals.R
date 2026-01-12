#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ape)
  library(ggplot2)
  library(dplyr)
  library(glue)
})

# ----------------------------
# Top-level folders
# ----------------------------
base_dir <- "/home/rohan/PLM-Phylo/Results/Trees/MSAT_Trees/OMAM20_4mammalsAligned"
output_dir <- "/home/rohan/PLM-Phylo/Plots/"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# ----------------------------
# Configurable species rules
# ----------------------------
HUMAN <- "Homo_sapiens"
EXPECTED_SISTER <- "Macaca_fascicularis"
OTHER_SPECIES <- c("Rattus_norvegicus", "Monodelphis_domestica")

# Matrix dimensions
N_LAYERS <- 12
N_HEADS  <- 12

# ----------------------------
# Helpers: determine human's sister tip
# ----------------------------
human_sister <- function(tree, human_label) {
  ih <- match(human_label, tree$tip.label)
  if (is.na(ih)) return(NA_character_)
  
  ph <- tree$edge[tree$edge[, 2] == ih, 1]
  if (length(ph) != 1) return(NA_character_)
  
  children <- tree$edge[tree$edge[, 1] == ph, 2]
  tip_children <- children[children <= Ntip(tree)]
  tip_children <- tip_children[tip_children != ih]
  
  if (length(tip_children) == 1) return(tree$tip.label[tip_children])
  NA_character_
}

# Return categorical status: "correct", "wrong", "unknown"
human_status <- function(tree, human_label, expected_sister, other_species = NULL) {
  if (!is.null(other_species)) {
    expected_taxa <- c(human_label, expected_sister, other_species)
    if (length(setdiff(expected_taxa, tree$tip.label)) > 0) return("unknown")
  }
  
  sis <- human_sister(tree, human_label)
  if (is.na(sis)) return("unknown")
  if (sis == expected_sister) return("correct")
  "wrong"
}

# ----------------------------
# Check ASTRAL consensus tree for a domain
# ----------------------------
check_astral <- function(domain) {
  astral_file <- file.path(base_dir, domain, glue("{domain}_astral_output.nwk"))
  if (!file.exists(astral_file)) {
    return(list(status = "unknown", file = astral_file))
  }
  
  tree <- tryCatch(read.tree(astral_file), error = function(e) NULL)
  if (is.null(tree)) {
    return(list(status = "unknown", file = astral_file))
  }
  
  tree_u <- unroot(tree)
  st <- human_status(tree_u, HUMAN, EXPECTED_SISTER, OTHER_SPECIES)
  
  list(status = st, file = astral_file)
}

# ----------------------------
# Process one domain:
# - save matrix PNG
# - return %correct across layer-heads for that domain
# ----------------------------
process_domain <- function(domain) {
  input_dir <- glue("{base_dir}/{domain}/Trees/aDist_Trees/")
  if (!dir.exists(input_dir)) {
    message("[SKIP] Missing aDist_Trees dir for domain: ", domain)
    return(NULL)
  }
  
  tree_files <- list.files(input_dir, pattern = "\\.nwk$", full.names = TRUE)
  if (length(tree_files) == 0) {
    message("[SKIP] No .nwk files for domain: ", domain)
    return(NULL)
  }
  
  # Grid initialized to unknown (so missing layer/heads show as yellow)
  grid_df <- expand.grid(
    Head  = 1:N_HEADS,
    Layer = 1:N_LAYERS
  ) %>%
    mutate(Status = "unknown")
  
  for (tree_file in tree_files) {
    m <- regmatches(tree_file, regexec("layer(\\d+)_head(\\d+)", tree_file))[[1]]
    if (length(m) < 3) next
    
    layer <- as.integer(m[2])
    head  <- as.integer(m[3])
    if (is.na(layer) || is.na(head) || layer < 1 || layer > N_LAYERS || head < 1 || head > N_HEADS) next
    
    tree <- tryCatch(read.tree(tree_file), error = function(e) NULL)
    if (is.null(tree)) next
    
    tree_u <- unroot(tree)
    st <- human_status(tree_u, HUMAN, EXPECTED_SISTER, OTHER_SPECIES)
    
    grid_df$Status[grid_df$Layer == layer & grid_df$Head == head] <- st
  }
  
  # % correct PER DOMAIN across layer-heads: correct / (correct + wrong)
  n_correct <- sum(grid_df$Status == "correct", na.rm = TRUE)
  n_wrong   <- sum(grid_df$Status == "wrong",   na.rm = TRUE)
  denom     <- n_correct + n_wrong
  pct_correct_domain <- if (denom > 0) 100 * (n_correct / denom) else NA_real_
  
  # Plot matrix (correct/wrong/unknown)
  status_colors <- c(correct = "lightgreen", wrong = "red", unknown = "yellow")
  
  p <- ggplot(grid_df, aes(x = Layer, y = Head, fill = Status)) +
    geom_tile(color = "white", linewidth = 0.6) +
    scale_fill_manual(values = status_colors, drop = FALSE) +
    scale_x_continuous(
      breaks = 1:N_LAYERS,
      labels = paste0("Layer ", 1:N_LAYERS),
      expand = c(0, 0),
      position = "top"
    ) +
    scale_y_continuous(
      breaks = 1:N_HEADS,
      labels = paste0("Head ", 1:N_HEADS),
      expand = c(0, 0)
    ) +
    coord_fixed() +
    theme_minimal(base_size = 14) +
    theme(
      legend.position  = "none",
      panel.grid       = element_blank(),
      axis.title       = element_blank(),
      axis.ticks       = element_blank(),
      axis.text.x      = element_text(color = "black", margin = margin(b = 6)),
      axis.text.y      = element_text(color = "black", margin = margin(r = 6)),
      plot.background  = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      plot.title       = element_text(hjust = 0.5)
    ) +
    ggtitle(glue("{domain}  |  % correct: {ifelse(is.na(pct_correct_domain), 'NA', round(pct_correct_domain, 1))}%"))
  
  output_file <- file.path(output_dir, glue("{domain}_layerhead_correct_matrix.png"))
  ggsave(output_file, plot = p, width = 10, height = 10, units = "in", dpi = 300, bg = "white")
  
  # Print only per-domain % correct
  message(glue("{domain}: % correct (layerheads) = {ifelse(is.na(pct_correct_domain), 'NA', round(pct_correct_domain, 3))}  (correct={n_correct}, wrong={n_wrong})"))
  
  list(
    domain = domain,
    pct_correct = pct_correct_domain,
    n_correct = n_correct,
    n_wrong = n_wrong,
    output_file = output_file
  )
}

# ----------------------------
# Iterate all domains
# ----------------------------
domains <- list.dirs(base_dir, full.names = FALSE, recursive = FALSE)
domains <- domains[domains != ""]  # safety

domain_stats <- data.frame(
  Domain=character(),
  pct_correct=numeric(),
  correct=integer(),
  wrong=integer(),
  astral_status=character(),
  stringsAsFactors=FALSE
)

for (d in domains) {
  res <- process_domain(d)
  if (!is.null(res)) {
    # ASTRAL check (runs regardless of whether matrix succeeded, but easiest here)
    astr <- check_astral(d)
    message(glue("  {d}: ASTRAL consensus = {astr$status}"))
    
    domain_stats <- rbind(
      domain_stats,
      data.frame(
        Domain = res$domain,
        pct_correct = res$pct_correct,
        correct = res$n_correct,
        wrong = res$n_wrong,
        astral_status = astr$status,
        stringsAsFactors = FALSE
      )
    )
  } else {
    # Still report ASTRAL if present even if aDist trees missing
    astr <- check_astral(d)
    message(glue("{d}: (no layerhead trees) ASTRAL consensus = {astr$status}"))
    
    domain_stats <- rbind(
      domain_stats,
      data.frame(
        Domain = d,
        pct_correct = NA_real_,
        correct = NA_integer_,
        wrong = NA_integer_,
        astral_status = astr$status,
        stringsAsFactors = FALSE
      )
    )
  }
}

# ----------------------------
# Summary across domains (proteins)
# ----------------------------
if (nrow(domain_stats) > 0) {
  pcs <- domain_stats$pct_correct
  mean_pc <- mean(pcs, na.rm = TRUE)
  median_pc <- median(pcs, na.rm = TRUE)
  
  message("\n==============================")
  message("SUMMARY ACROSS DOMAINS")
  message("==============================")
  message(glue("Domains listed: {nrow(domain_stats)}"))
  message(glue("Domains with layerhead PC: {sum(!is.na(pcs))}"))
  message(glue("Mean % correct across domains:   {round(mean_pc, 3)}"))
  message(glue("Median % correct across domains: {round(median_pc, 3)}"))
  
  # ASTRAL totals
  astral_correct <- sum(domain_stats$astral_status == "correct", na.rm = TRUE)
  astral_wrong   <- sum(domain_stats$astral_status == "wrong", na.rm = TRUE)
  astral_unknown <- sum(domain_stats$astral_status == "unknown", na.rm = TRUE)
  
  message("\n==============================")
  message("ASTRAL CONSENSUS SUMMARY")
  message("==============================")
  message(glue("Correct consensus trees:   {astral_correct}"))
  message(glue("Wrong consensus trees:     {astral_wrong}"))
  message(glue("Unknown consensus trees:   {astral_unknown}"))
  
  # Save a CSV summary table (handy)
  csv_file <- file.path(output_dir, "DOMAIN_SUMMARY_WITH_ASTRAL.csv")
  write.csv(domain_stats, csv_file, row.names = FALSE)
  message("Saved summary table: ", csv_file)
  
  # ----------------------------
  # Boxplot: PCs across domains (one value per domain)
  # ----------------------------
  domain_stats_clean <- domain_stats %>% filter(!is.na(pct_correct))
  
  if (nrow(domain_stats_clean) > 0) {
    box <- ggplot(domain_stats_clean, aes(x = pct_correct, y = "")) +
      geom_boxplot(outlier.size = 2, linewidth = 1.0) +
      scale_x_continuous(
        limits = c(0, 100),
        breaks = seq(0, 100, 10),
        expand = expansion(mult = c(0.01, 0.03))
      ) +
      theme_bw(base_size = 22) +
      theme(
        plot.background  = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA),
        panel.grid.major = element_line(linewidth = 0.6),
        panel.grid.minor = element_line(linewidth = 0.4),
        plot.margin      = margin(t = 8, r = 28, b = 10, l = 10),
        
        axis.title.y     = element_blank(),
        axis.text.y      = element_blank(),
        axis.ticks.y     = element_blank(),
        
        axis.title.x     = element_text(size = 24, color = "black"),
        axis.text.x      = element_text(size = 20, color = "black"),
        
        axis.ticks       = element_line(linewidth = 0.8),
        axis.line        = element_line(linewidth = 0.9),
        
        plot.title       = element_blank()
      ) +
      xlab("% correct layerheads\nper protein")
    
    box_file <- file.path(output_dir, "PC_BOXPLOT_ACROSS_DOMAINS.png")
    ggsave(box_file, plot = box, width = 5, height = 5, units = "in", dpi = 300, bg = "white")
    message("Saved boxplot: ", box_file)
  } else {
    message("No PC data available for boxplot (all NA).")
  }
  
} else {
  message("No domains found.")
}
