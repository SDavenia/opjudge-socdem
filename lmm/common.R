# Shared helpers for the LMM analyses of the gender setups (candidate, reference, both).
#
# Data: scores/hb_{setup}_gender/{judge}_by_prompt.csv, one row per scored prompt
# (reference x candidate x level x prompt variation). The cue levels of an item are minimal pairs:
# the same (reference, candidate) opinions with only the identity phrase changed.

suppressPackageStartupMessages({
  library(data.table)
  library(lme4)
  library(lmerTest)
  library(emmeans)
})

# Sum-to-zero coding: main effects are averaged over the other factors (e.g. the gender effect over
# the 5 prompts), and type III tests are meaningful. Gender contrasts come from emmeans, so the
# coding does not change them.
options(contrasts = c("contr.sum", "contr.poly"))
emm_options(lmer.df = "satterthwaite", lmerTest.limit = 1e5, pbkrtest.limit = 0)

LEVELS <- c("neutral", "female", "male")
MATCHES <- c("same", "partial", "opposite")
PROMPTS <- c("base", "lex_var", "reverse_order", "lex_var_reverse_order", "criteria_after")

# As analyze.py: same (agree-agree, ...), partial (one of the two neutral), opposite (agree-disagree).
stance_match <- function(reference, candidate) {
  fifelse(reference == candidate, "same",
          fifelse(reference == "neutral" | candidate == "neutral", "partial", "opposite"))
}

# score_dir: one directory, or several separated by commas (a joint analysis, e.g. the hb and vp ones;
# columns one of them lacks are NA for its rows).
load_scores <- function(score_dir, exclude_mismatch = FALSE) {
  dirs <- strsplit(score_dir, ",", fixed = TRUE)[[1]]
  files <- list.files(dirs, pattern = "_by_prompt\\.csv$", full.names = TRUE)
  if (!length(files)) stop("no *_by_prompt.csv in ", score_dir)
  d <- rbindlist(lapply(files, fread), fill = TRUE)
  d <- d[!is.na(score)]
  if (exclude_mismatch) d <- d[is.na(stance_label_mismatch) | stance_label_mismatch == FALSE]
  d[, `:=`(
    judge = factor(judge_model),
    situation = factor(situation_id),
    reference = factor(reference_original_text_id),
    candidate = factor(candidate_original_text_id),
    # The item whose cue levels are minimal pairs.
    base_pair = factor(paste(reference_original_text_id, candidate_original_text_id, sep = "__")),
    prompt = factor(prompt_variation, PROMPTS),
    match = factor(stance_match(reference_stance_group, candidate_stance_group), MATCHES),
    ref_stance = factor(reference_stance_group, c("agree", "disagree", "neutral")),
    authority = factor(expected_authority, c("M", "W"))
  )]
  d[]
}

# Fit, then drop the random terms estimated at (numerically) zero and refit. Each term is a
# variance component (no correlated slopes), so dropping one at 0 does not change the fit and only
# removes the singularity warning. Returns the refitted model.
fit_pruned <- function(fixed, random_terms, data, tol = 1e-4, REML = TRUE) {
  ctrl <- lmerControl(calc.derivs = FALSE, optimizer = "bobyqa")
  repeat {
    f <- reformulate(c(fixed, sprintf("(1 | %s)", random_terms)), response = "score")
    m <- suppressMessages(lmer(f, data = data, REML = REML, control = ctrl))
    vc <- as.data.frame(VarCorr(m))
    vc <- vc[vc$grp != "Residual", ]
    zero <- vc$grp[vc$sdcor < tol * sigma(m)]
    if (!length(zero) || length(zero) == length(random_terms)) return(m)
    message("  dropping zero random terms: ", paste(zero, collapse = ", "))
    random_terms <- setdiff(random_terms, zero)
  }
}

# Share of the total score variance per source: each fixed term's variance of its contribution to
# the linear predictor (X_t b_t, Nakagawa-style partition of the marginal R2) and each random
# variance component. Fixed terms are almost orthogonal under sum coding in this near-balanced design,
# so their covariances are left out (`fixed_covariance` gives what is left out).
variance_partition <- function(m) {
  X <- getME(m, "X")
  b <- fixef(m)
  assign <- attr(X, "assign")
  terms <- attr(terms(m), "term.labels")
  fixed <- sapply(seq_along(terms), function(i) {
    cols <- which(assign == i)
    var(as.vector(X[, cols, drop = FALSE] %*% b[cols]))
  })
  total_fixed <- var(as.vector(X[, -1, drop = FALSE] %*% b[-1]))
  vc <- as.data.frame(VarCorr(m))
  out <- rbind(
    data.table(source = terms, kind = "fixed", variance = fixed),
    data.table(source = "(fixed covariance)", kind = "fixed", variance = total_fixed - sum(fixed)),
    data.table(source = vc$grp, kind = fifelse(vc$grp == "Residual", "residual", "random"),
               variance = vc$vcov)
  )
  out[, share := variance / sum(variance)]
  out[order(-share)]
}

anova_table <- function(m) {
  a <- anova(m, type = 3, ddf = "Satterthwaite")
  data.table(term = rownames(a), num_df = a$NumDF, den_df = a$DenDF, F = a$`F value`, p = a$`Pr(>F)`)
}

# emmeans contrasts as a table. `weights = "proportional"` averages over the other factors with the
# observed cell frequencies, i.e. the same pooling as the raw means of analyze.py (e.g. the pooled
# gap weights same/partial/opposite pairs by how many there are).
contrast_table <- function(emm, method, ...) {
  ct <- contrast(emm, method = method, ...)
  s <- as.data.table(summary(ct, infer = c(TRUE, TRUE)))
  setnames(s, c("lower.CL", "upper.CL", "p.value"), c("lower", "upper", "p"), skip_absent = TRUE)
  s
}

GENDER_CONTRASTS <- list(
  "female - neutral" = c(-1, 1, 0),
  "male - neutral" = c(-1, 0, 1),
  "female - male" = c(0, 1, -1)
)

write_txt <- function(path, ...) cat(..., file = path, sep = "", append = TRUE)
