#!/usr/bin/env Rscript
# LMMs for the candidate intervention (gender cue on the candidate, original references).
#
#   Rscript lmm/candidate.R [score_dir] [out_dir] [--exclude_mismatch] [--no_pooled]
#   Rscript lmm/candidate.R scores/hb_candidate_gender analysis/hb/candidate_gender/lmm
#
# Two models:
#
# 1. Per judge (one model each):
#      score ~ level * match + level * prompt + level * authority + match * prompt
#              + (1 | situation) + (1 | reference) + (1 | candidate) + (1 | base_pair)
#              + (1 | base_pair:prompt) + (1 | situation:level) + (1 | base_pair:level)
#    level     candidate cue: neutral / female / male (within base_pair: the minimal pairs)
#    match     stance match of reference and candidate: same / partial / opposite (between pairs)
#    prompt    the 5 prompt variations (within base_pair)
#    authority expected_authority of the situation: M (cars, sport, technology) / W (childcare)
#    Random effects: situation, the reference and candidate opinions (candidate = situation x stance),
#    the base pair, how the pair reacts to the prompt (base_pair:prompt) and to the cue
#    (base_pair:level, situation:level). The last two are the error terms of the gender effects:
#    without them the 5 prompts of a pair would count as independent replicates of its gender gap.
#    Residual = pair x level x prompt.
#
# 2. Pooled over judges: the same fixed effects crossed with judge, the random effects nested in
#    judge where a judge can react differently (base_pair:judge, base_pair:judge:prompt,
#    base_pair:level:judge, situation:level:judge). Tests whether the gender effects differ
#    between judges (level:judge, level:match:judge, ...) and gives the average over judges.
#
# Outputs in out_dir: variance_partition.csv (what moves the score most), anova.csv (type III F,
# Satterthwaite df), contrasts.csv (gender gaps pooled, by match, by prompt, by authority, and the
# congruence contrast), random_effects.csv, summary.txt, models.rds.

source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), "common.R"))

args <- commandArgs(TRUE)
flags <- args[startsWith(args, "--")]
pos <- args[!startsWith(args, "--")]
score_dir <- if (length(pos) >= 1) pos[1] else "scores/hb_candidate_gender"
out_dir <- if (length(pos) >= 2) pos[2] else "analysis/hb/candidate_gender/lmm"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
summary_path <- file.path(out_dir, "summary.txt")
if (file.exists(summary_path)) file.remove(summary_path)

d <- load_scores(score_dir, exclude_mismatch = "--exclude_mismatch" %in% flags)
stopifnot(all(d$reference_level == "original"))
d[, level := factor(candidate_level, LEVELS)]

FIXED <- "level * match + level * prompt + level * authority + match * prompt"
RANDOM <- c("situation", "reference", "candidate", "base_pair", "base_pair:prompt",
            "situation:level", "base_pair:level")
FIXED_POOLED <- sprintf("judge * (%s)", FIXED)
RANDOM_POOLED <- c("situation", "reference", "candidate", "base_pair", "base_pair:judge",
                   "base_pair:judge:prompt", "situation:level", "situation:level:judge",
                   "base_pair:level", "base_pair:level:judge")

# Gender gaps of a fitted model: pooled (by = NULL) or within each level of factor `by`. In the pooled
# model they are averaged over judges.
gender_contrasts <- function(m, by = NULL) {
  spec <- if (is.null(by)) ~ level else reformulate(sprintf("level | %s", by))
  ct <- contrast_table(emmeans(m, spec, weights = "proportional"), GENDER_CONTRASTS)
  ct[, by := if (is.null(by)) "pooled" else by]
  ct
}

# Authority-balanced congruence: ((f - m | W) - (f - m | M)) / 2, as analyze.py's
# "congruent - incongruent". Cells of level x authority, level fastest.
congruence <- function(m) {
  emm <- emmeans(m, ~ level * authority, weights = "proportional")
  ct <- contrast_table(emm, list("congruent - incongruent" = c(0, -0.5, 0.5, 0, 0.5, -0.5)))
  ct[, by := "authority"]
  ct
}

all_contrasts <- function(m) {
  rbindlist(c(lapply(list(NULL, "match", "prompt", "authority"), gender_contrasts, m = m),
              list(congruence(m))), fill = TRUE)
}

random_table <- function(m) {
  vc <- as.data.table(as.data.frame(VarCorr(m)))[, .(group = grp, sd = sdcor, variance = vcov)]
  vc[, n_levels := sapply(group, function(g) if (g == "Residual") nobs(m) else nlevels(getME(m, "flist")[[g]]))]
  vc
}

models <- list()
partition <- anovas <- contrasts <- randoms <- list()

for (j in levels(d$judge)) {
  message("fitting ", j)
  dj <- droplevels(d[judge == j])
  m <- fit_pruned(FIXED, RANDOM, dj)
  models[[j]] <- m
  partition[[j]] <- variance_partition(m)[, judge := j]
  anovas[[j]] <- anova_table(m)[, judge := j]
  contrasts[[j]] <- all_contrasts(m)[, judge := j]
  randoms[[j]] <- random_table(m)[, judge := j]
}

if (!"--no_pooled" %in% flags) {
  message("fitting pooled model (all judges)")
  m <- fit_pruned(FIXED_POOLED, RANDOM_POOLED, d)
  models[["pooled"]] <- m
  partition[["pooled"]] <- variance_partition(m)[, judge := "pooled"]
  anovas[["pooled"]] <- anova_table(m)[, judge := "pooled"]
  # Average over judges (judges weighted equally: same number of rows each).
  contrasts[["pooled"]] <- all_contrasts(m)[, judge := "pooled"]
  randoms[["pooled"]] <- random_table(m)[, judge := "pooled"]
}

partition <- rbindlist(partition)
anovas <- rbindlist(anovas)
contrasts <- rbindlist(contrasts, fill = TRUE)
randoms <- rbindlist(randoms)
fwrite(partition, file.path(out_dir, "variance_partition.csv"))
fwrite(anovas, file.path(out_dir, "anova.csv"))
fwrite(contrasts, file.path(out_dir, "contrasts.csv"))
fwrite(randoms, file.path(out_dir, "random_effects.csv"))
saveRDS(models, file.path(out_dir, "models.rds"))

# Human-readable summary.
fmt <- function(x, k = 3) formatC(x, format = "f", digits = k)
fmt_p <- function(p) ifelse(p < 1e-4, "<1e-4", formatC(p, format = "g", digits = 2))
write_txt(summary_path, "LMMs, candidate intervention: ", score_dir, "\n",
          "fixed: ", FIXED, "\n\n")
for (j in names(models)) {
  m <- models[[j]]
  write_txt(summary_path, "==== ", j, " (n = ", nobs(m), ", REML ", fmt(REMLcrit(m), 1),
            if (isSingular(m)) ", singular" else "", ")\n",
            "random: ", paste(names(getME(m, "flist")), collapse = ", "), "\n\n",
            "Variance partition (share of total score variance):\n")
  p <- partition[judge == j][share >= 0.001]
  write_txt(summary_path, sprintf("  %-28s %-8s %6.1f%%\n", p$source, p$kind, 100 * p$share), "\n")
  write_txt(summary_path, "Type III F tests:\n")
  a <- anovas[judge == j]
  write_txt(summary_path, sprintf("  %-22s F(%d, %7.1f) = %8.2f  p = %s\n",
                                  a$term, as.integer(a$num_df), a$den_df, a$F, fmt_p(a$p)), "\n")
  write_txt(summary_path, "Gender contrasts (estimate [95% CI], p):\n")
  ct <- contrasts[judge == j]
  group_col <- intersect(c("match", "prompt", "authority"), names(ct))
  ct[, group := do.call(paste, c(lapply(group_col, function(g) fifelse(is.na(get(g)), "", as.character(get(g)))), sep = ""))]
  write_txt(summary_path, sprintf("  %-10s %-22s %-24s %+.3f [%+.3f, %+.3f]  p = %s\n",
                                  ct$by, ct$group, ct$contrast, ct$estimate, ct$lower, ct$upper,
                                  fmt_p(ct$p)), "\n")
}
message("wrote ", out_dir)
