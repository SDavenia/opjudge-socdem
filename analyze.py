"""
First analysis of a score directory: how much does the judge's score move when only the prompt
variation changes, and how much when only the sociodemographic level changes?

Reads every {judge}_by_prompt.csv of a directory written by score.py (e.g. scores/reference_gender).
An item is a pair of original opinions (reference_original_text_id x candidate_original_text_id);
its versions differ by prompt variation and by level of the intervened side.

Unsigned variation, all on the same scale so that they can be compared:
  mad     mean absolute difference |score_a - score_b| over all the pairs of conditions of an
          item (it does not grow with the number of conditions, unlike the std: 5 prompts vs 2-3
          levels), averaged over the items
  changed share of those pairs whose scores differ at all
computed for three factors:
  prompt   across the prompt variations, level fixed                      (unit: item x level)
  levels   across the non-neutral levels (female vs male), prompt fixed   (unit: item x prompt)
  cue      each non-neutral level vs neutral (naming a group at all)      (unit: item x prompt)

Signed gaps, prompt fixed (unit: item x prompt): each level - neutral, and each pair of non-neutral
levels (female - male), overall and by prompt variation.

--relative: the same measures and plots, as differences relative to a baseline score, per situation.
For each situation: its mean difference / its mean baseline (e.g. mean female score 0.15 vs mean neutral
0.10 -> +50%); the pooled value is the unweighted mean of those situation percentages (by prompt
variation: the same, within that prompt). Not a mean of per-pair ratios, which would divide by the many
0 scores. Baselines:
  mad (unsigned)   the mean of the compared scores (prompt: the 5 prompts; levels: female and male;
                   cue: each level and neutral)
  signed gaps      the neutral score (or, with no neutral level, the subtracted level's), so that
                   female - male = (female - neutral) - (male - neutral) holds in relative terms too
  changed          unchanged (already scale-free)
Situations with a low baseline get large percentages and weigh as much as any other in the mean.
The CIs bootstrap the situations; the per-situation SEs are delta-method (linearized) SEs of the ratio.
Files get a "_relative" suffix, so both versions can sit in the same directory.

95% CIs: bootstrap over situations (the pairs of a situation share its opinions, so they are not
independent). The sociodemographic variation also contains the numerical noise of vLLM batching
(about 3% of the answers change between two layouts, see README), which a rerun would estimate.

Output, in {out_dir} (default analysis/{name of results_dir}/):
  variation.csv              judge x factor: mad and changed, with CIs
  signed_gaps.csv            judge x contrast x prompt variation ("all" = pooled): mean gap, CI
  mean_scores.csv            judge x level x prompt variation: mean score
  variation.png/.pdf         the three factors, by judge
  signed_gaps.png/.pdf       the gaps, by judge
  signed_gaps_by_prompt.png/.pdf  the gaps, by prompt variation
  situations.csv             judge x situation x measure (each signed gap, levels/cue mad and changed):
                             mean over the situation's observations, SE clustered by reference
  top_situations.csv         situations sorted by |female - male| gap averaged over the judges, with
                             each judge's gap, how many judges agree on its sign, and the unsigned mad
  situation_agreement.csv    Spearman correlation of the per-situation gaps between judges
  top_situations.png/.pdf    the --top situations of top_situations.csv
  top_situations_max.csv/.png      the same, ranked by the largest |gap| of any judge (judge_of_max says whose)
  model_specific_top_situations/top_situations_{judge}.csv/.png  the same, ranked by that judge's |gap|,
                             plotted for that judge only
  topics.csv                 judge x grouping (topic domain, subdomain) x topic x measure (each signed gap,
                             levels/cue mad): mean and CI over the topic's situations, n_situations
  topic_heterogeneity.csv    judge x grouping x measure: do the topics differ more than random groups of
                             situations? eta2 (between-topic share of the variance of the situation values),
                             its value under no topic effect, (G - 1) / (N - 1), and a permutation p-value
  topics_{domain,subdomain}.png/.pdf  the signed gaps by topic (groups of >= --min_situations situations)
The topics need the topic columns of topics.py in the scores. Groupings: selection_topic, expected_authority,
topic_domain, topic_subdomain, those present with >= 2 values.
  stances.csv                judge x grouping (reference stance, stance pair ref -> cand, stance match) x group x
                             measure: mean and CI over the situations (CSV column `topic` = the group)
  stance_heterogeneity.csv   do the stance groups differ? eta2 of the item means vs stances shuffled within situations
  stances_{reference,pair,match}.png/.pdf  the signed gaps by stance group
  model_specific_top_situations_by_stance/top_situations_{judge}_by_{match,pair}.csv/.png  each judge's top
                             situations (ranked as in model_specific_top_situations), with the gap of each stance
                             group (match: same/partial/opposite; pair: the 9 ref -> cand stances) side by side,
                             each from the situation's items in that group only (n {group} = its references there)
Stereotype congruence (signed gap "congruent - incongruent", see STEREOTYPE_COLUMNS): expected level - the other,
with the stereotype's groups weighted equally, = ((f - m | W) - (f - m | M)) / 2 for gender.

Usage:
    python analyze.py scores/reference_gender
    python analyze.py scores/reference_gender --out_dir analysis/pilot --n_boot 2000
    python analyze.py scores/reference_gender --relative     # -> *_relative.{csv,png,pdf}
    python analyze.py scores/hb_reference_gender scores/vp_reference_gender --out_dir analysis/joint/reference_gender
                                                             # joint: the datasets pooled, + a "dataset" grouping
"""
import argparse
import textwrap
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from add_sociodem_preamble import NEUTRAL_LEVEL, SOCIODEMOGRAPHICS  # noqa: E402
from judges import REGISTRY  # noqa: E402
from prompts import SCORING_PROMPTS  # noqa: E402
from score import ORIGINAL_LEVEL  # noqa: E402

ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis"
ITEM_KEYS = ["reference_original_text_id", "candidate_original_text_id"]

# Stereotype congruence: per variable, the situation column saying which level the stereotype expects
# (e.g. who is expected to have more authority on the topic) and its values -> levels. Gives the signed
# contrast CONGRUENT = expected level - the other level (only with exactly two non-neutral levels), pooled
# with the expected levels weighted equally: then a gap shared by all the situations (e.g. female > male
# everywhere) cancels out, and CONGRUENT = ((f - m | W) - (f - m | M)) / 2 for gender.
STEREOTYPE_COLUMNS = {"gender": ("expected_authority", {"M": "male", "W": "female"})}
CONGRUENT = "congruent - incongruent"

# Judges keep their color and marker whatever subset is plotted: slots in the order of
# judges.REGISTRY (then OpenRouter judges, alphabetically). Categorical palette of the dataviz
# reference (light mode); the marker is the secondary encoding past 3 judges.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
INK, INK_2, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dirs", type=Path, nargs="+",
                        help="Directories of score.py, e.g. scores/hb_reference_gender; several (e.g. the hb and vp "
                             "ones of a setup) are analysed jointly.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Default: analysis/{run_name}.")
    parser.add_argument("--judges", nargs="+", default=None, help="Only these judges (default: every file).")
    parser.add_argument("--n_boot", type=int, default=1000, help="Bootstrap resamples of the situations.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=15, help="Situations to print and plot.")
    parser.add_argument("--n_perm", type=int, default=2000, help="Permutations of the topic heterogeneity test.")
    parser.add_argument("--min_situations", type=int, default=5,
                        help="Topic groups with fewer situations are left out of the topic plots (not the CSV).")
    parser.add_argument("--relative", action="store_true",
                        help="Differences relative to a baseline score (see above); files get a _relative suffix.")
    return parser.parse_args()


# ---------------------------------------------------------------------------------------------
# Loading

def as_dirs(results_dirs):
    return [results_dirs] if isinstance(results_dirs, (str, Path)) else list(results_dirs)


def run_name(results_dirs):
    """The name of one directory, or of several sharing a setup: hb_reference_gender + vp_reference_gender
    -> hb+vp_reference_gender."""
    names = [Path(d).resolve().name for d in as_dirs(results_dirs)]
    if len(names) == 1:
        return names[0]
    prefixes, setups = zip(*(n.split("_", 1) if "_" in n else (n, "") for n in names))
    return f"{'+'.join(prefixes)}_{setups[0]}" if len(set(setups)) == 1 else "+".join(names)


def default_out_dir(name):
    """analysis/<dataset>/<setup>: hb_reference_gender -> analysis/hb/reference_gender, hb+vp_... ->
    analysis/joint/..."""
    prefix, _, setup = name.partition("_")
    if prefix in ("hb", "vp"):
        return ANALYSIS_DIR / prefix / setup
    if set(prefix.split("+")) == {"hb", "vp"}:
        return ANALYSIS_DIR / "joint" / setup
    return ANALYSIS_DIR / name


def load(results_dirs, judges=None):
    """The *_by_prompt.csv of one or more directories (the datasets of a joint analysis: columns one of them
    lacks are NaN for its rows)."""
    frames = []
    for results_dir in as_dirs(results_dirs):
        for path in sorted(Path(results_dir).glob("*_by_prompt.csv")):
            judge = path.name.removesuffix("_by_prompt.csv")
            if judges and judge not in judges:
                continue
            df = pd.read_csv(path, low_memory=False)
            df["judge"] = judge
            frames.append(df)
    if not frames:
        raise SystemExit(f"No *_by_prompt.csv in {results_dirs}")
    df = pd.concat(frames, ignore_index=True)

    # The level of a version: of the intervened side, or of both sides ("ref|cand").
    sides = [s for s in ("reference", "candidate") if (df[f"{s}_level"] != ORIGINAL_LEVEL).any()]
    if len(sides) == 1:
        df["level"] = df[f"{sides[0]}_level"]
    elif len(sides) == 2:
        df["level"] = df["reference_level"] + "|" + df["candidate_level"]
    else:
        df["level"] = ORIGINAL_LEVEL
    df["item"] = df[ITEM_KEYS[0]] + "__" + df[ITEM_KEYS[1]]
    variable = df["variable"].dropna().iloc[0] if "variable" in df and df["variable"].notna().any() else None
    return df, variable, sides


def judge_order(judges):
    registry = list(REGISTRY)
    return sorted(judges, key=lambda j: (registry.index(j) if j in registry else len(registry), j))


def style_of(judge, all_judges):
    i = judge_order(all_judges).index(judge) % len(PALETTE)
    return PALETTE[i], MARKERS[i]


def ordered_levels(levels, variable):
    """Non-neutral levels in the order of SOCIODEMOGRAPHICS, then any other, sorted."""
    known = list(SOCIODEMOGRAPHICS.get(variable, {}).get("levels", {}))
    others = [lv for lv in levels if lv != NEUTRAL_LEVEL]
    return sorted(others, key=lambda lv: (known.index(lv) if lv in known else len(known), lv))


# ---------------------------------------------------------------------------------------------
# Metrics

def pairwise(wide):
    """Mean |a - b| and share of a != b over the pairs of columns of `wide`, per row (NaN-aware)."""
    values = wide.to_numpy(dtype=float)
    diffs = [values[:, i] - values[:, j] for i, j in combinations(range(values.shape[1]), 2)]
    diffs = np.stack(diffs, axis=1)
    valid = ~np.isnan(diffs)
    n = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mad = np.where(valid, np.abs(diffs), 0).sum(axis=1) / n
        changed = np.where(valid, diffs != 0, 0).sum(axis=1) / n
    return pd.DataFrame({"mad": mad, "changed": changed}, index=wide.index)


def bootstrap_draws(values, clusters, n_boot, rng, denominators=None):
    """Mean of `values`, the bootstrap distribution of that mean (resampling whole clusters) and the
    number of observations.

    With `denominators`, a relative mean: each cluster's mean value divided by its mean denominator (e.g.
    a situation's mean gap / its mean neutral score), then the unweighted mean of those cluster ratios.
    Clusters whose mean denominator is 0 are dropped."""
    values = np.asarray(values, dtype=float)
    relative = denominators is not None
    denominators = np.asarray(denominators, dtype=float) if relative else np.ones_like(values)
    ok = ~np.isnan(values) & ~np.isnan(denominators)
    values, denominators, clusters = values[ok], denominators[ok], np.asarray(clusters)[ok]
    if len(values) == 0:
        return np.nan, None, 0
    codes, uniques = pd.factorize(clusters)
    sums = np.bincount(codes, weights=values, minlength=len(uniques))
    den_sums = np.bincount(codes, weights=denominators, minlength=len(uniques))
    if relative:
        ratios = sums[den_sums != 0] / den_sums[den_sums != 0]
        weights = rng.multinomial(len(ratios), np.full(len(ratios), 1 / len(ratios)), size=n_boot)
        return ratios.mean(), weights @ ratios / len(ratios), len(values)
    weights = rng.multinomial(len(uniques), np.full(len(uniques), 1 / len(uniques)), size=n_boot)
    return values.mean(), (weights @ sums) / (weights @ den_sums), len(values)


def cluster_bootstrap(values, clusters, n_boot, rng, denominators=None):
    """Mean of `values` and the 95% CI of a bootstrap that resamples whole clusters (see bootstrap_draws)."""
    mean, boot, n = bootstrap_draws(values, clusters, n_boot, rng, denominators)
    if boot is None:
        return np.nan, np.nan, np.nan, 0
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, lo, hi, n


def stratified_bootstrap(values, clusters, strata, n_boot, rng, denominators=None):
    """The unweighted mean over the strata of each stratum's mean (e.g. M- and W-authority situations weigh
    the same, whatever their numbers), with the 95% CI of a bootstrap resampling the clusters within each
    stratum. Observations with no stratum are left out."""
    values = np.asarray(values, dtype=float)
    strata = np.asarray(strata, dtype=object)
    has = pd.notna(strata)
    means, boots, n = [], [], 0
    for stratum in pd.unique(strata[has]):
        mask = has & (strata == stratum)
        mean, boot, k = bootstrap_draws(values[mask], np.asarray(clusters)[mask], n_boot, rng,
                                        None if denominators is None else np.asarray(denominators)[mask])
        if boot is not None:
            means.append(mean)
            boots.append(boot)
            n += k
    if not means:
        return np.nan, np.nan, np.nan, 0
    lo, hi = np.percentile(np.mean(boots, axis=0), [2.5, 97.5])
    return np.mean(means), lo, hi, n


def summarize(values, clusters, n_boot, rng, denominators=None, strata=None, **keys):
    if strata is None:
        mean, lo, hi, n = cluster_bootstrap(values, clusters, n_boot, rng, denominators)
    else:
        mean, lo, hi, n = stratified_bootstrap(values, clusters, strata, n_boot, rng, denominators)
    return {**keys, "mean": mean, "ci_low": lo, "ci_high": hi, "n": n}


def analyze_judge(df, variable, n_boot, rng, relative=False):
    judge = df["judge"].iloc[0]
    variation, gaps = [], []

    # Prompt: across the prompt variations, level fixed. Relative: to the mean over the prompts.
    by_prompt = df.pivot_table(index=["situation_id", "item", "level"], columns="prompt_variation",
                               values="score", aggfunc="first")
    prompt = pairwise(by_prompt)
    situations = prompt.index.get_level_values("situation_id")
    for metric in ("mad", "changed"):
        base = by_prompt.mean(axis=1) if relative and metric == "mad" else None
        variation.append(summarize(prompt[metric], situations, n_boot, rng, base,
                                   judge=judge, factor="prompt", metric=metric))

    levels = ordered_levels(df["level"].unique(), variable)
    if df["level"].nunique() < 2:
        return variation, gaps, None

    by_level = level_table(df)
    situations = by_level.index.get_level_values("situation_id")
    prompts = by_level.index.get_level_values("prompt_variation")

    signed, unsigned = level_measures(by_level, levels, relative, expected_levels(df, variable))

    # Unsigned: levels (female vs male) and cue (each level vs neutral), prompt fixed.
    for name, (values, base) in unsigned.items():
        factor, metric = name.split(" ")
        variation.append(summarize(values, situations, n_boot, rng, base, judge=judge, factor=factor, metric=metric))

    # Signed gaps, pooled and by prompt variation. The congruence contrast weighs the stereotype's
    # levels the same (stratified bootstrap), so that a gap shared by all the situations cancels out.
    expected = expected_levels(df, variable)
    strata_all = None if expected is None else expected.reindex(situations).to_numpy()
    for name, (gap, base) in signed.items():
        strata = strata_all if name == CONGRUENT else None
        gaps.append(summarize(gap, situations, n_boot, rng, base, strata,
                              judge=judge, contrast=name, prompt_variation="all"))
        for pv in SCORING_PROMPTS:
            mask = np.asarray(prompts == pv)
            if mask.any():
                gaps.append(summarize(gap[mask], situations[mask], n_boot, rng,
                                      None if base is None else base[mask],
                                      None if strata is None else strata[mask],
                                      judge=judge, contrast=name, prompt_variation=pv))
    # Gaps first, then the unsigned measures (the order of the per-situation table).
    measures = {**signed, **unsigned}
    by_situation = per_situation(pd.DataFrame({m: v for m, (v, _) in measures.items()}), judge,
                                 {m: b for m, (_, b) in measures.items() if b is not None})
    return variation, gaps, by_situation


def level_table(df):
    """Scores with one column per level; a row per observation (situation, reference, item, prompt)."""
    return df.pivot_table(index=["situation_id", "reference_original_text_id", "item", "prompt_variation"],
                          columns="level", values="score", aggfunc="first")


def expected_levels(df, variable):
    """The level the stereotype expects in each situation (Series by situation_id), or None."""
    if variable not in STEREOTYPE_COLUMNS or STEREOTYPE_COLUMNS[variable][0] not in df:
        return None
    column, mapping = STEREOTYPE_COLUMNS[variable]
    return df.drop_duplicates("situation_id").set_index("situation_id")[column].map(mapping)


def level_measures(by_level, levels, relative, expected=None):
    """The per-observation measures of the levels, as (signed, unsigned): dicts measure -> (values, baseline),
    the baseline being None in absolute mode or for `changed` (already scale-free).

    signed    each level - neutral and each pair of levels (female - male); baseline: the neutral score
              (or, with no neutral level, the subtracted level's). With `expected` (the stereotype's level
              of each situation) and two levels, also CONGRUENT: expected level - the other one
    unsigned  "levels mad/changed" across the non-neutral levels, "cue mad/changed" each level vs neutral
              averaged over the levels; baseline: the mean of the compared scores"""
    has_neutral = NEUTRAL_LEVEL in by_level.columns

    def denominator(metric, baseline):
        return baseline if relative and metric != "changed" else None

    signed, unsigned = {}, {}
    if len(levels) >= 2:
        between = pairwise(by_level[levels])
        levels_mean = by_level[levels].mean(axis=1)
        for metric in ("mad", "changed"):
            unsigned[f"levels {metric}"] = (between[metric], denominator(metric, levels_mean))
    if has_neutral and levels:
        diffs = by_level[levels].sub(by_level[NEUTRAL_LEVEL], axis=0)
        # Baseline: the mean of the compared scores, averaged over the (level, neutral) pairs.
        cue_mean = by_level[levels].add(by_level[NEUTRAL_LEVEL], axis=0).div(2).mean(axis=1)
        unsigned["cue mad"] = (diffs.abs().mean(axis=1), denominator("mad", cue_mean))
        unsigned["cue changed"] = ((diffs != 0).where(diffs.notna()).mean(axis=1), None)

    contrasts = {f"{lv} - {NEUTRAL_LEVEL}": (lv, NEUTRAL_LEVEL) for lv in levels} if has_neutral else {}
    contrasts.update({f"{a} - {b}": (a, b) for a, b in combinations(levels, 2)})
    for name, (a, b) in contrasts.items():
        base = by_level[NEUTRAL_LEVEL] if has_neutral else by_level[b]
        signed[name] = (by_level[a] - by_level[b], denominator("gap", base))
    if expected is not None and len(levels) == 2 and expected.notna().any():
        a, b = levels
        exp = expected.reindex(by_level.index.get_level_values("situation_id")).to_numpy()
        sign = np.where(exp == a, 1.0, np.where(exp == b, -1.0, np.nan))
        base = by_level[NEUTRAL_LEVEL] if has_neutral else by_level[b]
        signed[CONGRUENT] = ((by_level[a] - by_level[b]) * sign, denominator("gap", base))
    return signed, unsigned


# ---------------------------------------------------------------------------------------------
# Topics

# Grouping name -> situation column. Only those present with >= 2 values are used: the manual selection
# (selection_topic, expected_authority) and the Habermas taxonomy (topic_domain, topic_subdomain).
# "dataset" only appears in joint analyses (hb + vp).
TOPIC_GROUPINGS = {"selection": "selection_topic", "authority": "expected_authority",
                   "domain": "topic_domain", "subdomain": "topic_subdomain", "dataset": "dataset"}


def available_groupings(df):
    """The groupings whose column is known for every situation (not, in a joint analysis, the topic columns
    of a single dataset) and has >= 2 values."""
    situations = df.drop_duplicates("situation_id")
    return {g: c for g, c in TOPIC_GROUPINGS.items()
            if c in df and situations[c].notna().all() and situations[c].nunique() >= 2}


def situation_values(values, baseline, situations):
    """Each situation's mean value, or, with a baseline, its mean value / its mean baseline (NaN if 0)."""
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float), "situation": np.asarray(situations)})
    frame["base"] = 1.0 if baseline is None else np.asarray(baseline, dtype=float)
    sums = frame.dropna().groupby("situation")[["value", "base"]].sum()
    return sums["value"] / sums["base"].replace(0, np.nan)


def heterogeneity(per_situation_values, groups, n_perm, rng):
    """Do the groups (topics) differ more than random groups of the same sizes would? On the situation
    values: eta2 = between-group sum of squares / total sum of squares, its permutation p-value (shuffling
    the groups of the situations) and its expected value under no topic effect, (G - 1) / (N - 1)."""
    ok = per_situation_values.notna().to_numpy()
    values = per_situation_values.to_numpy()[ok]
    codes, uniques = pd.factorize(np.asarray(groups)[ok])
    counts = np.bincount(codes).astype(float)
    total = ((values - values.mean()) ** 2).sum()

    def between(c):
        means = np.bincount(c, weights=values, minlength=len(uniques)) / counts
        return (counts * (means - values.mean()) ** 2).sum()

    observed = between(codes)
    null = np.array([between(rng.permutation(codes)) for _ in range(n_perm)])
    return {"eta2": observed / total, "eta2_null": (len(uniques) - 1) / (len(values) - 1),
            "p_value": (1 + (null >= observed).sum()) / (1 + n_perm),
            "n_groups": len(uniques), "n_situations": len(values)}


def analyze_topics(df, variable, groupings, n_boot, n_perm, rng, relative=False):
    """Every signed gap and unsigned mad by each grouping of the situations (mean and bootstrap CI over the
    situations of the group, absolute or relative as in the pooled analysis), and a permutation test of
    whether they differ between the groups."""
    levels = ordered_levels(df["level"].unique(), variable)
    by_level = level_table(df)
    signed, unsigned = level_measures(by_level, levels, relative)
    # Not the congruence contrast: within a stereotype group it is +-(female - male), and between groups
    # it differs by construction.
    measures = {**signed, **{m: v for m, v in unsigned.items() if m.endswith("mad")}}
    return topic_breakdown(df, by_level, measures, groupings, n_boot, n_perm, rng)


def topic_breakdown(df, by_level, measures, groupings, n_boot, n_perm, rng):
    """`measures` (name -> (per-observation values of by_level, baseline or None)) by each situation grouping:
    mean and CI over the group's situations, and the heterogeneity test of the groups."""
    judge = df["judge"].iloc[0]
    situations = by_level.index.get_level_values("situation_id")
    topic_of = df.drop_duplicates("situation_id").set_index("situation_id")

    rows, tests = [], []
    for grouping, column in groupings.items():
        groups = topic_of[column].reindex(situations).to_numpy()
        situation_group = topic_of[column]
        for measure, (values, base) in measures.items():
            values = values.to_numpy()
            base = None if base is None else base.to_numpy()
            for group in sorted(pd.unique(groups)):
                mask = groups == group
                rows.append({**summarize(values[mask], situations[mask], n_boot, rng,
                                         None if base is None else base[mask],
                                         judge=judge, grouping=grouping, topic=group, measure=measure),
                             "n_situations": pd.unique(situations[mask]).size})
            per_sit = situation_values(values, base, situations)
            tests.append({"judge": judge, "grouping": grouping, "measure": measure,
                          **heterogeneity(per_sit, situation_group.reindex(per_sit.index), n_perm, rng)})
    return rows, tests


# ---------------------------------------------------------------------------------------------
# Stances

STANCES = ["agree", "neutral", "disagree"]
STANCE_COLUMNS = ["reference_stance_group", "candidate_stance_group"]


def stance_match(reference, candidate):
    """same (agree-agree, ...), opposite (agree-disagree), partial (one of the two neutral)."""
    reference, candidate = np.asarray(reference, dtype=object), np.asarray(candidate, dtype=object)
    return np.where(reference == candidate, "same",
                    np.where((reference == "neutral") | (candidate == "neutral"), "partial", "opposite"))


# Grouping name -> (labels from the reference and candidate stances, order of the groups).
STANCE_GROUPINGS = {
    "reference": (lambda r, c: np.asarray(r, dtype=object), STANCES),
    "pair": (lambda r, c: np.char.add(np.char.add(np.asarray(r, dtype=str), " -> "), np.asarray(c, dtype=str)),
             [f"{r} -> {c}" for r in STANCES for c in STANCES]),
    "match": (stance_match, ["same", "partial", "opposite"]),
}


def within_permutation(clusters, rng):
    """Indices that shuffle the elements of an array within each cluster."""
    codes = pd.factorize(np.asarray(clusters))[0]
    base = np.argsort(codes, kind="stable")
    shuffled = np.lexsort((rng.random(len(codes)), codes))
    perm = np.empty(len(codes), dtype=int)
    perm[base] = shuffled
    return perm


def analyze_stances(df, variable, n_boot, n_perm, rng, relative=False):
    """Every signed gap (with the congruence contrast) and unsigned mad by the stance of the reference, by the
    pair of stances (reference -> candidate) and by their match: mean and CI by bootstrap over the
    situations, absolute or relative as in the pooled analysis.

    The stances vary within the situations, so the test of whether the groups differ (absolute values only)
    keeps the situations: unit = item (reference x candidate, mean over the prompts), statistic = eta2
    between the groups, null = the stances of the references, and of the candidates, shuffled within each
    situation (situation effects and composition, e.g. M/W authority, stay the same)."""
    levels = ordered_levels(df["level"].unique(), variable)
    by_level = level_table(df)
    situations = by_level.index.get_level_values("situation_id")
    expected = expected_levels(df, variable)
    signed, unsigned = level_measures(by_level, levels, relative, expected)
    measures = {**signed, **{m: v for m, v in unsigned.items() if m.endswith("mad")}}
    strata = {} if expected is None else {CONGRUENT: expected.reindex(situations).to_numpy()}
    return stance_breakdown(df, by_level, measures, n_boot, n_perm, rng, relative, strata)


def stance_breakdown(df, by_level, measures, n_boot, n_perm, rng, relative=False, strata=None):
    """`measures` (name -> (per-observation values of by_level, baseline or None)) by the stance groupings, with
    the within-situation permutation test (absolute mode only; see analyze_stances). `strata`: measure ->
    per-observation stratum, for the measures pooled over strata with equal weights (stratified bootstrap)."""
    judge = df["judge"].iloc[0]
    situations = by_level.index.get_level_values("situation_id")
    strata = strata or {}

    stance_of = df.drop_duplicates("item").set_index("item")[STANCE_COLUMNS]
    items = by_level.index.get_level_values("item")
    ref_stance = stance_of["reference_stance_group"].reindex(items).to_numpy()
    cand_stance = stance_of["candidate_stance_group"].reindex(items).to_numpy()

    # Items for the test: their situation, reference, candidate, stances, and mean value of each measure.
    unit = pd.DataFrame({m: np.asarray(v, dtype=float) for m, (v, _) in measures.items()},
                        index=by_level.index).groupby(level=["situation_id", "reference_original_text_id",
                                                             "item"]).mean()
    unit_items = unit.index.get_level_values("item")
    unit_sit = unit.index.get_level_values("situation_id").to_numpy()
    unit_ref = unit.index.get_level_values("reference_original_text_id").to_numpy()
    unit_cand = df.drop_duplicates("item").set_index("item")["candidate_original_text_id"].reindex(unit_items)
    unit_ref_stance = stance_of["reference_stance_group"].reindex(unit_items).to_numpy()
    unit_cand_stance = stance_of["candidate_stance_group"].reindex(unit_items).to_numpy()
    # One permutation draws new stances for the references and candidates of each situation.
    refs = pd.DataFrame({"sit": unit_sit, "ref": unit_ref, "stance": unit_ref_stance}).drop_duplicates("ref")
    cands = pd.DataFrame({"sit": unit_sit, "cand": unit_cand.to_numpy(), "stance": unit_cand_stance}
                         ).drop_duplicates(["sit", "cand"])
    ref_pos = pd.Index(refs["ref"]).get_indexer(unit_ref)
    cand_pos = pd.MultiIndex.from_frame(cands[["sit", "cand"]]).get_indexer(
        pd.MultiIndex.from_arrays([unit_sit, unit_cand.to_numpy()]))
    perms = [(refs["stance"].to_numpy()[within_permutation(refs["sit"], rng)][ref_pos],
              cands["stance"].to_numpy()[within_permutation(cands["sit"], rng)][cand_pos]) for _ in range(n_perm)]

    def eta2(values, labels):
        codes, uniques = pd.factorize(labels)
        counts = np.bincount(codes, minlength=len(uniques))
        means = np.bincount(codes, weights=values, minlength=len(uniques)) / counts
        return (counts * (means - values.mean()) ** 2).sum() / ((values - values.mean()) ** 2).sum()

    rows, tests = [], []
    for grouping, (label_of, order) in STANCE_GROUPINGS.items():
        groups = label_of(ref_stance, cand_stance)
        for measure, (values, base) in measures.items():
            values = np.asarray(values, dtype=float)
            base = None if base is None else np.asarray(base, dtype=float)
            for group in [g for g in order if (groups == g).any()]:
                mask = groups == group
                rows.append({**summarize(values[mask], situations[mask], n_boot, rng,
                                         None if base is None else base[mask],
                                         strata[measure][mask] if measure in strata else None,
                                         judge=judge, grouping=grouping, topic=group, measure=measure),
                             "n_situations": pd.unique(situations[mask]).size})
            if relative:
                continue
            y = unit[measure].to_numpy()
            ok = ~np.isnan(y)
            observed = eta2(y[ok], label_of(unit_ref_stance, unit_cand_stance)[ok])
            null = np.array([eta2(y[ok], label_of(r, c)[ok]) for r, c in perms])
            tests.append({"judge": judge, "grouping": grouping, "measure": measure, "eta2": observed,
                          "eta2_null_mean": null.mean(), "p_value": (1 + (null >= observed).sum()) / (1 + n_perm),
                          "n_groups": len(pd.unique(label_of(unit_ref_stance, unit_cand_stance)[ok])),
                          "n_items": int(ok.sum())})
    return rows, tests


def stance_situations(df, variable, relative=False):
    """The signed gaps of each situation (as per_situation) within each stance group of the "match" and "pair"
    groupings: only the situation's items whose stances fall in the group. Long format, with the columns
    `grouping` and `group`."""
    judge = df["judge"].iloc[0]
    levels = ordered_levels(df["level"].unique(), variable)
    by_level = level_table(df)
    signed, _ = level_measures(by_level, levels, relative)
    stance_of = df.drop_duplicates("item").set_index("item")[STANCE_COLUMNS]
    items = by_level.index.get_level_values("item")
    ref_stance = stance_of["reference_stance_group"].reindex(items).to_numpy()
    cand_stance = stance_of["candidate_stance_group"].reindex(items).to_numpy()
    frames = []
    for grouping in ("match", "pair"):
        label_of, order = STANCE_GROUPINGS[grouping]
        groups = label_of(ref_stance, cand_stance)
        for group in [g for g in order if (groups == g).any()]:
            mask = groups == group
            observed = pd.DataFrame({m: v[mask] for m, (v, _) in signed.items()})
            baselines = {m: b[mask] for m, (_, b) in signed.items() if b is not None}
            frames.append(per_situation(observed, judge, baselines).assign(grouping=grouping, group=group))
    return pd.concat(frames, ignore_index=True) if frames else None


def per_situation(observed, judge, baselines=None):
    """Each measure averaged over the observations (item x prompt) of each situation, with its standard
    error clustered by reference opinion (each reference has the same number of candidates and prompts,
    so the mean of the reference means is the mean of the observations). Long format.

    With `baselines` (measure -> baseline score of each observation), those measures are relative: the
    situation's mean divided by its mean baseline, with the delta-method SE of that ratio (the SE of the
    reference means of the linearized values (x - ratio * baseline) / mean baseline)."""
    keys = ["situation_id", "reference_original_text_id"]
    by_reference = observed.groupby(level=keys).mean()
    grouped = by_reference.groupby(level="situation_id")
    mean, n = grouped.mean(), grouped.size()
    se = grouped.std().div(np.sqrt(n), axis=0)
    situation_of = by_reference.index.get_level_values("situation_id")
    for measure, baseline in (baselines or {}).items():
        base_by_reference = baseline.groupby(level=keys).mean().reindex(by_reference.index)
        base_mean = base_by_reference.groupby(level="situation_id").mean().replace(0, np.nan)
        ratio = mean[measure] / base_mean
        linearized = ((by_reference[measure] - ratio.reindex(situation_of).to_numpy() * base_by_reference)
                      / base_mean.reindex(situation_of).to_numpy())
        mean[measure] = ratio
        se[measure] = linearized.groupby(level="situation_id").std() / np.sqrt(n)
    out = (mean.stack().rename("mean").to_frame()
           .join(se.stack().rename("se"), how="left")
           .rename_axis(["situation_id", "measure"]).reset_index())
    out["n_references"] = out["situation_id"].map(n)
    out.insert(0, "judge", judge)
    return out


def top_situations(by_situation, situations_text, judges, contrast, expected=None, rank_by="mean"):
    """One row per situation: the gap of `contrast` for each judge, its mean over the judges, its largest
    absolute value over the judges, and the mean unsigned difference between the levels. Sorted by
    `rank_by`: "mean" = |mean gap over the judges|, "max" = the largest |gap| of any judge, or a judge's
    name = that judge's |gap|. With `expected`, also the level the stereotype expects in the situation."""
    gap = by_situation[by_situation["measure"] == contrast]
    wide = gap.pivot(index="situation_id", columns="judge", values="mean").reindex(columns=judges)
    se = gap.pivot(index="situation_id", columns="judge", values="se").reindex(columns=judges)
    out = pd.DataFrame(index=wide.index)
    out["situation"] = situations_text.reindex(wide.index)
    if expected is not None:
        out["expected_level"] = expected.reindex(wide.index)
    out["n_references"] = gap.groupby("situation_id")["n_references"].first()
    out["gap_mean_over_judges"] = wide.mean(axis=1)
    out["n_judges_same_sign"] = (np.sign(wide).eq(np.sign(out["gap_mean_over_judges"]), axis=0)).sum(axis=1)
    out["gap_max_abs_over_judges"] = wide.abs().max(axis=1)
    out["judge_of_max"] = wide.abs().idxmax(axis=1)
    for judge in judges:
        out[f"gap {judge}"] = wide[judge]
        out[f"se {judge}"] = se[judge]
    unsigned = by_situation[by_situation["measure"] == "levels mad"]
    if len(unsigned):
        out["levels_mad_mean_over_judges"] = unsigned.groupby("situation_id")["mean"].mean()
    key = {"mean": out["gap_mean_over_judges"].abs(), "max": out["gap_max_abs_over_judges"]}.get(rank_by)
    if key is None:
        key = wide[rank_by].abs()
    out = out.reindex(key.sort_values(ascending=False).index)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out.reset_index()


# ---------------------------------------------------------------------------------------------
# Plots

def setup_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def percent_axis(axis):
    axis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))


def save(fig, out_dir, name, relative=False):
    name += "_relative" if relative else ""
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def figure(**kwargs):
    fig, axes = plt.subplots(**kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, np.atleast_1d(axes)


def legend(fig, judges, all_judges, y=1.02):
    handles = [plt.Line2D([], [], color=style_of(j, all_judges)[0], marker=style_of(j, all_judges)[1],
                          linestyle="", markersize=7, label=j) for j in judges]
    fig.legend(handles=handles, loc="upper center", ncol=len(judges), frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.5, y))


def plot_variation(variation, factor_labels, judges, out_dir, title, relative=False):
    mad_label = ("Mean absolute score difference,\n% of the mean of the compared scores" if relative
                 else "Mean absolute score difference")
    metrics = [("mad", mad_label), ("changed", "Share of score pairs that differ")]
    factors = [f for f in factor_labels if f in set(variation["factor"])]
    fig, axes = figure(ncols=2, figsize=(11, 4.2))
    width = 0.8 / len(judges)
    for ax, (metric, ylabel) in zip(axes, metrics):
        setup_axes(ax)
        ax.grid(axis="x", visible=False)
        sub = variation[variation["metric"] == metric]
        for k, judge in enumerate(judges):
            color, _ = style_of(judge, judges)
            rows = sub[sub["judge"] == judge].set_index("factor").reindex(factors)
            x = np.arange(len(factors)) + (k - (len(judges) - 1) / 2) * width
            ax.bar(x, rows["mean"], width=width * 0.9, color=color, label=judge, zorder=2)
            ax.errorbar(x, rows["mean"], yerr=[rows["mean"] - rows["ci_low"], rows["ci_high"] - rows["mean"]],
                        fmt="none", ecolor=INK, elinewidth=1, capsize=0, zorder=3)
        ax.set_xticks(np.arange(len(factors)), [factor_labels[f] for f in factors], fontsize=9, color=INK)
        ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
        if metric == "changed":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        elif relative:
            percent_axis(ax.yaxis)
    legend(fig, judges, judges)
    fig.suptitle(title, fontsize=11, color=INK, y=1.09)
    fig.tight_layout()
    save(fig, out_dir, "variation", relative)


def gap_label(relative, baseline, sep=" "):
    """Axis label of the signed gaps; `baseline` names the score they are relative to."""
    return f"Mean score difference,{sep}% of the mean {baseline} score" if relative else "Mean score difference"


def plot_gaps(gaps, judges, out_dir, title, relative=False, baseline=NEUTRAL_LEVEL, name="signed_gaps"):
    pooled = gaps[gaps["prompt_variation"] == "all"]
    contrasts = list(dict.fromkeys(pooled["contrast"]))
    fig, axes = figure(figsize=(7, 0.9 + 0.75 * len(contrasts) * max(1, len(judges) / 3)))
    ax = axes[0]
    setup_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=1)
    step = 0.6 / max(1, len(judges) - 1) if len(judges) > 1 else 0
    for k, judge in enumerate(judges):
        color, marker = style_of(judge, judges)
        rows = pooled[pooled["judge"] == judge].set_index("contrast").reindex(contrasts)
        y = np.arange(len(contrasts)) + (k - (len(judges) - 1) / 2) * step
        ax.errorbar(rows["mean"], y, xerr=[rows["mean"] - rows["ci_low"], rows["ci_high"] - rows["mean"]],
                    fmt="none", ecolor=color, elinewidth=2, capsize=0, zorder=2)
        ax.scatter(rows["mean"], y, color=color, marker=marker, s=48, edgecolors=SURFACE, linewidths=1.5,
                   zorder=3, label=judge)
    ax.set_yticks(np.arange(len(contrasts)), contrasts, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel(f"{gap_label(relative, baseline)} (95% CI, bootstrap over situations)", fontsize=9, color=INK_2)
    if relative:
        percent_axis(ax.xaxis)
    legend(fig, judges, judges)
    fig.suptitle(title, fontsize=11, color=INK, y=1.1)
    fig.tight_layout()
    save(fig, out_dir, name, relative)


def plot_gaps_by_prompt(gaps, judges, out_dir, title, relative=False, baseline=NEUTRAL_LEVEL,
                        name="signed_gaps_by_prompt"):
    by_prompt = gaps[gaps["prompt_variation"] != "all"]
    contrasts = list(dict.fromkeys(by_prompt["contrast"]))
    prompts = [p for p in SCORING_PROMPTS if p in set(by_prompt["prompt_variation"])]
    fig, axes = figure(ncols=len(contrasts), figsize=(4.2 * len(contrasts), 4.2), sharey=True)
    step = 0.5 / max(1, len(judges) - 1) if len(judges) > 1 else 0
    for ax, contrast in zip(axes, contrasts):
        setup_axes(ax)
        ax.grid(axis="x", visible=False)
        ax.axhline(0, color=INK_MUTED, linewidth=1, zorder=1)
        for k, judge in enumerate(judges):
            color, marker = style_of(judge, judges)
            rows = (by_prompt[(by_prompt["judge"] == judge) & (by_prompt["contrast"] == contrast)]
                    .set_index("prompt_variation").reindex(prompts))
            x = np.arange(len(prompts)) + (k - (len(judges) - 1) / 2) * step
            ax.errorbar(x, rows["mean"], yerr=[rows["mean"] - rows["ci_low"], rows["ci_high"] - rows["mean"]],
                        fmt="none", ecolor=color, elinewidth=2, capsize=0, zorder=2)
            ax.scatter(x, rows["mean"], color=color, marker=marker, s=40, edgecolors=SURFACE, linewidths=1.5,
                       zorder=3)
        ax.set_xticks(np.arange(len(prompts)), [p.replace("_", "\n") for p in prompts], fontsize=8, color=INK)
        ax.set_title(contrast, fontsize=10, color=INK)
    axes[0].set_ylabel(f"{gap_label(relative, baseline, sep=chr(10))} (95% CI)", fontsize=9, color=INK_2)
    if relative:
        percent_axis(axes[0].yaxis)
    legend(fig, judges, judges)
    fig.suptitle(title, fontsize=11, color=INK, y=1.1)
    fig.tight_layout()
    save(fig, out_dir, name, relative)


def plot_top_situations(top, judges, contrast, n_top, out_dir, title, relative=False, baseline=NEUTRAL_LEVEL,
                        name="top_situations", plot_judges=None, pooled=None,
                        pooled_label="the judge's mean gap over all the situations"):
    """The first `n_top` rows of `top`, with the gap of each of `plot_judges` (default: all the judges; the
    colors stay those of `judges`). No legend for a single judge (the title names it). `pooled`: judge ->
    its mean gap over all the situations, drawn as a dotted line (labelled when a single judge is plotted)."""
    rows = top.head(n_top)
    all_judges, judges = judges, plot_judges or judges
    fig, axes = figure(figsize=(10, 1.0 + 0.55 * len(rows) * max(1, len(judges) / 3)))
    ax = axes[0]
    setup_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=1)
    step = 0.6 / max(1, len(judges) - 1) if len(judges) > 1 else 0
    for judge in judges if pooled else []:
        if judge in pooled and np.isfinite(pooled[judge]):
            ax.axvline(pooled[judge], color=style_of(judge, all_judges)[0], linewidth=1.2, linestyle=":", zorder=1)
            if len(judges) == 1:
                value = f"{pooled[judge]:+.1%}" if relative else f"{pooled[judge]:+.4f}"
                ax.annotate(f"{judge} mean: {value}", xy=(pooled[judge], 1), xycoords=("data", "axes fraction"),
                            xytext=(4, 2), textcoords="offset points", fontsize=8, color=INK_2, va="bottom")
    for k, judge in enumerate(judges):
        color, marker = style_of(judge, all_judges)
        y = np.arange(len(rows)) + (k - (len(judges) - 1) / 2) * step
        ax.errorbar(rows[f"gap {judge}"], y, xerr=1.96 * rows[f"se {judge}"].fillna(0), fmt="none", ecolor=color,
                    elinewidth=2, capsize=0, zorder=2)
        ax.scatter(rows[f"gap {judge}"], y, color=color, marker=marker, s=40, edgecolors=SURFACE, linewidths=1.5,
                   zorder=3)
    tag = (lambda r: f"[{r['expected_level']}] ") if "expected_level" in rows else (lambda r: "")
    labels = [f"{r['rank']}. {tag(r)}{textwrap.shorten(str(r['situation']), 70)}  (n={r['n_references']})"
              for _, r in rows.iterrows()]
    ax.set_yticks(np.arange(len(rows)), labels, fontsize=8, color=INK)
    ax.invert_yaxis()
    what = (f"Mean {contrast} score difference in the situation,\n% of its mean {baseline} score" if relative
            else f"Mean {contrast} score difference in the situation")
    dotted = f"\ndotted: {pooled_label}" if pooled else ""
    ax.set_xlabel(f"{what}\n(±1.96 SE over its references; n = references){dotted}", fontsize=9, color=INK_2)
    if relative:
        percent_axis(ax.xaxis)
    # Fixed room (in inches) at the top for the title and the legend, whatever the figure's height.
    height = fig.get_figheight()
    fig.suptitle(title, fontsize=11, color=INK, y=1 - 0.15 / height, va="top")
    if len(judges) > 1:
        legend(fig, judges, all_judges, y=1 - 0.45 / height)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.8 if len(judges) > 1 else 0.5) / height))
    save(fig, out_dir, name, relative)


STANCE_COLORS = {"agree": PALETTE[2], "neutral": PALETTE[3], "disagree": PALETTE[7]}
STANCE_MARKERS = {"agree": "^", "neutral": "o", "disagree": "v"}
MATCH_STYLES = {"same": (PALETTE[2], "o"), "partial": (PALETTE[3], "s"), "opposite": (PALETTE[7], "D")}


def stance_group_style(group):
    """match groups: a color each; pair groups (ref -> cand): the candidate stance's color and marker (the
    reference stance is the panel)."""
    if group in MATCH_STYLES:
        return MATCH_STYLES[group]
    candidate = group.split(" -> ")[1]
    return STANCE_COLORS[candidate], STANCE_MARKERS[candidate]


def stance_top_table(ranked, by_stance, judge, order):
    """The situations of `ranked` (a judge's top situations, in its order) with, for each stance group, the
    judge's gap, SE and number of references in that group (wide: "gap {group}", "se {group}", "n {group}")."""
    keep = ["situation_id", "rank", "situation"] + (["expected_level"] if "expected_level" in ranked else [])
    out = ranked[keep + ["n_references", f"gap {judge}", f"se {judge}"]].rename(
        columns={f"gap {judge}": "gap all", f"se {judge}": "se all", "n_references": "n all"})
    wide = by_stance.pivot(index="situation_id", columns="group", values=["mean", "se", "n_references"])
    for group in [g for g in order if g in set(by_stance["group"])]:
        for stat, col in (("gap", "mean"), ("se", "se"), ("n", "n_references")):
            out[f"{stat} {group}"] = out["situation_id"].map(wide[(col, group)])
    return out


def plot_stance_top_situations(table, judge, grouping, contrast, n_top, out_dir, title, relative, baseline, name,
                               group_means=None):
    """The first `n_top` situations of `table` (stance_top_table), one point (±1.96 SE) per stance group side by
    side within each situation's row; the judge's gap over all the situation's items as a grey bar, each group's
    mean gap over all the situations as a dotted line of its color. "pair": one panel per reference stance, the
    candidate stances within it."""
    rows = table.head(n_top)
    present = [c.removeprefix("gap ") for c in rows.columns if c.startswith("gap ") and c != "gap all"]
    if grouping == "pair":
        panels = [(f"reference: {r}", [g for g in present if g.startswith(f"{r} -> ")]) for r in STANCES]
        panels = [(t, gs) for t, gs in panels if gs]
    else:
        panels = [(None, present)]
    fig, axes = figure(ncols=len(panels), sharey=True, sharex=True,
                       figsize=(10 if len(panels) == 1 else 15, 1.0 + 0.32 * len(rows) * 2))
    y0 = np.arange(len(rows))
    for ax, (panel_title, groups) in zip(axes, panels):
        setup_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=1)
        ax.barh(y0, rows["gap all"], height=0.8, color=GRID, zorder=1)
        step = 0.6 / max(1, len(groups) - 1)
        for k, group in enumerate(groups):
            color, marker = stance_group_style(group)
            if group_means and np.isfinite(group_means.get(group, np.nan)):
                ax.axvline(group_means[group], color=color, linewidth=1, linestyle=":", zorder=1)
            y = y0 + (k - (len(groups) - 1) / 2) * step
            ax.errorbar(rows[f"gap {group}"], y, xerr=1.96 * rows[f"se {group}"].fillna(0), fmt="none",
                        ecolor=color, elinewidth=1.5, capsize=0, zorder=2)
            label = group.split(" -> ")[1] if grouping == "pair" else group
            ax.scatter(rows[f"gap {group}"], y, color=color, marker=marker, s=30, edgecolors=SURFACE,
                       linewidths=1, zorder=3, label=label)
        if panel_title:
            ax.set_title(panel_title, fontsize=10, color=INK)
        if relative:
            percent_axis(ax.xaxis)
    tag = (lambda r: f"[{r['expected_level']}] ") if "expected_level" in rows else (lambda r: "")
    labels = [f"{r['rank']}. {tag(r)}{textwrap.shorten(str(r['situation']), 70)}  (n={r['n all']})"
              for _, r in rows.iterrows()]
    axes[0].set_yticks(y0, labels, fontsize=8, color=INK)
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    what = (f"Mean {contrast} score difference in the situation, % of its mean {baseline} score" if relative
            else f"Mean {contrast} score difference in the situation")
    fig.supxlabel(f"{what}\n(±1.96 SE over its references in the group; grey bar: all the situation's items; "
                  f"dotted: the group's mean gap over all the situations; n = references)", fontsize=9, color=INK_2)
    height = fig.get_figheight()
    fig.suptitle(title, fontsize=11, color=INK, y=1 - 0.15 / height, va="top")
    by_label = dict(zip(*reversed(axes[0].get_legend_handles_labels())))
    for ax in axes[1:]:
        by_label.update(dict(zip(*reversed(ax.get_legend_handles_labels()))))
    fig.legend(list(by_label.values()), list(by_label.keys()), loc="upper center", ncol=len(by_label),
               frameon=False, fontsize=8, labelcolor=INK_2, bbox_to_anchor=(0.5, 1 - 0.4 / height),
               title="candidate stance" if grouping == "pair" else "stance match", title_fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.0 / height))
    save(fig, out_dir, name, relative)


def plot_topics(topics, gaps, judges, grouping, sort_by, min_situations, out_dir, title, relative=False,
                baseline=NEUTRAL_LEVEL, order=None, name=None, unit="topic"):
    """One panel per signed contrast, one row per topic group (with at least `min_situations` situations),
    sorted by the `sort_by` gap averaged over the judges (or in the given `order`); dotted lines: each
    judge's pooled gap. Saved as `name` (default topics_{grouping})."""
    pooled = gaps[gaps["prompt_variation"] == "all"]
    contrasts = list(dict.fromkeys(pooled["contrast"]))
    sub = topics[(topics["grouping"] == grouping) & (topics["n_situations"] >= min_situations)]
    contrasts = [c for c in contrasts if c in set(sub["measure"])]
    if order is None:
        order = (sub[sub["measure"] == sort_by].groupby("topic")["mean"].mean().sort_values(ascending=False).index)
    else:
        order = [g for g in order if g in set(sub["topic"])]
    if not len(order):
        return
    n_sit = sub.groupby("topic")["n_situations"].first()
    fig, axes = figure(ncols=len(contrasts), sharey=True,
                       figsize=(3.6 * len(contrasts) + 3.4, 1.4 + 0.36 * len(order) * max(1, len(judges) / 3)))
    step = 0.6 / max(1, len(judges) - 1) if len(judges) > 1 else 0
    for ax, contrast in zip(axes, contrasts):
        setup_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=1)
        for k, judge in enumerate(judges):
            color, marker = style_of(judge, judges)
            overall = pooled[(pooled["judge"] == judge) & (pooled["contrast"] == contrast)]["mean"]
            if len(overall):
                ax.axvline(overall.iloc[0], color=color, linewidth=1, linestyle=":", zorder=1)
            rows = sub[(sub["judge"] == judge) & (sub["measure"] == contrast)].set_index("topic").reindex(order)
            y = np.arange(len(order)) + (k - (len(judges) - 1) / 2) * step
            ax.errorbar(rows["mean"], y, xerr=[rows["mean"] - rows["ci_low"], rows["ci_high"] - rows["mean"]],
                        fmt="none", ecolor=color, elinewidth=1.6, capsize=0, zorder=2)
            ax.scatter(rows["mean"], y, color=color, marker=marker, s=30, edgecolors=SURFACE, linewidths=1.2,
                       zorder=3)
        ax.set_title(contrast, fontsize=10, color=INK)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        if relative:
            percent_axis(ax.xaxis)
    axes[0].set_yticks(np.arange(len(order)),
                       [f"{textwrap.shorten(str(t), 48)}  (n={n_sit[t]})" for t in order], fontsize=8, color=INK)
    axes[0].invert_yaxis()
    fig.supxlabel(f"{gap_label(relative, baseline)} (95% CI, bootstrap over the {unit}'s situations; "
                  f"dotted: pooled; n = situations)", fontsize=9, color=INK_2)
    height = fig.get_figheight()
    fig.suptitle(title, fontsize=11, color=INK, y=1 - 0.15 / height, va="top")
    legend(fig, judges, judges, y=1 - 0.45 / height)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / height))
    save(fig, out_dir, name or f"topics_{grouping}", relative)


def load_situation_texts(results_dirs):
    frames = [pd.read_csv(p, usecols=["situation_id", "situation"])
              for d in as_dirs(results_dirs) for p in Path(d).glob("*_mean.csv")]
    if not frames:
        return pd.Series(dtype=str)
    return pd.concat(frames).drop_duplicates("situation_id").set_index("situation_id")["situation"]


# ---------------------------------------------------------------------------------------------

def main():
    args = parse_args()
    name = run_name(args.results_dirs)
    out_dir = args.out_dir or default_out_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df, variable, sides = load(args.results_dirs, args.judges)
    judges = judge_order(df["judge"].unique())
    levels = ordered_levels(df["level"].unique(), variable)
    intervened = " & ".join(sides) or "no side"
    print(f"{name}: judges {judges}, variable {variable}, intervened {intervened}, levels "
          f"{levels + ([NEUTRAL_LEVEL] if NEUTRAL_LEVEL in set(df['level']) else [])}")

    variation, gaps, by_situation = [], [], []
    for judge in judges:
        v, g, s = analyze_judge(df[df["judge"] == judge], variable, args.n_boot, rng, args.relative)
        variation += v
        gaps += g
        if s is not None:
            by_situation.append(s)
    variation, gaps = pd.DataFrame(variation), pd.DataFrame(gaps)
    mean_scores = (df.groupby(["judge", "level", "prompt_variation"])["score"].mean().unstack()
                   .reindex(columns=[p for p in SCORING_PROMPTS if p in set(df["prompt_variation"])]))
    mean_scores["all"] = df.groupby(["judge", "level"])["score"].mean()

    rel = args.relative
    suffix = "_relative" if rel else ""
    # What the signed gaps are relative to (see analyze_judge).
    baseline = NEUTRAL_LEVEL if NEUTRAL_LEVEL in set(df["level"]) else "subtracted level's"
    variation.to_csv(out_dir / f"variation{suffix}.csv", index=False)
    gaps.to_csv(out_dir / f"signed_gaps{suffix}.csv", index=False)
    mean_scores.to_csv(out_dir / "mean_scores.csv")

    factor_labels = {"prompt": "Prompt variation\n(5 prompts, level fixed)",
                     "levels": f"{variable or 'Level'}: {' vs '.join(levels)}\n(prompt fixed)",
                     "cue": f"{variable or 'Level'} cue vs {NEUTRAL_LEVEL}\n(prompt fixed)"}
    title = f"{name}: {intervened} intervened" + (" (relative)" if rel else "")
    plot_variation(variation, factor_labels, judges, out_dir, f"Score variation by factor — {title}", rel)
    if len(gaps):
        plot_gaps(gaps, judges, out_dir, f"Signed score gaps — {title}", rel, baseline)
        plot_gaps_by_prompt(gaps, judges, out_dir, f"Signed score gaps by prompt variation — {title}", rel, baseline)

    pd.set_option("display.width", 250, "display.max_columns", 20)
    if rel:  # every value is a ratio or a share: print as percentages
        fmt = lambda r: f"{r['mean']:.2%} [{r['ci_low']:.2%}, {r['ci_high']:.2%}]"  # noqa: E731
    else:
        fmt = lambda r: f"{r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"  # noqa: E731
    unit = " (mad relative to the mean of the compared scores)" if rel else ""
    table = variation.assign(value=variation.apply(fmt, axis=1)).pivot(
        index=["judge", "metric"], columns="factor", values="value")
    print(f"\nUnsigned variation{unit} (mean [95% CI])\n",
          table.reindex(columns=[f for f in factor_labels if f in table]))
    if len(gaps):
        pooled = gaps[gaps["prompt_variation"] == "all"]
        print(f"\nSigned gaps, pooled over prompts{f' (relative to the {baseline} score)' if rel else ''} "
              "(mean [95% CI])\n",
              pooled.assign(value=pooled.apply(fmt, axis=1)).pivot(index="judge", columns="contrast", values="value"))
    print("\nMean score by level\n", mean_scores["all"].unstack().round(4))

    # Situations: where the gap between the levels is largest.
    if by_situation:
        by_situation = pd.concat(by_situation, ignore_index=True)
        contrast = f"{levels[0]} - {levels[1]}" if len(levels) >= 2 else gaps["contrast"].iloc[0]
        expected = expected_levels(df, variable)
        top = top_situations(by_situation, load_situation_texts(args.results_dirs), judges, contrast, expected)
        by_situation.to_csv(out_dir / f"situations{suffix}.csv", index=False)
        top.to_csv(out_dir / f"top_situations{suffix}.csv", index=False)
        plot_top_situations(top, judges, contrast, args.top, out_dir,
                            f"Situations with the largest {contrast} gap (mean over judges) — {title}", rel, baseline)
        texts = load_situation_texts(args.results_dirs)
        pooled = (gaps[(gaps["contrast"] == contrast) & (gaps["prompt_variation"] == "all")]
                  .set_index("judge")["mean"].to_dict())
        top_max = top_situations(by_situation, texts, judges, contrast, expected, rank_by="max")
        top_max.to_csv(out_dir / f"top_situations_max{suffix}.csv", index=False)
        plot_top_situations(top_max, judges, contrast, args.top, out_dir,
                            f"Situations with the largest {contrast} gap (max |gap| over judges) — {title}", rel,
                            baseline, name="top_situations_max", pooled=pooled)
        judge_dir = out_dir / "model_specific_top_situations"
        judge_dir.mkdir(exist_ok=True)
        for judge in judges:
            top_judge = top_situations(by_situation, texts, judges, contrast, expected, rank_by=judge)
            top_judge.to_csv(judge_dir / f"top_situations_{judge}{suffix}.csv", index=False)
            plot_top_situations(top_judge, judges, contrast, args.top, judge_dir,
                                f"Situations with the largest {contrast} gap for {judge} — {title}", rel, baseline,
                                name=f"top_situations_{judge}", plot_judges=[judge], pooled=pooled)

        gap_by_judge = (by_situation[by_situation["measure"] == contrast]
                        .pivot(index="situation_id", columns="judge", values="mean").reindex(columns=judges))
        agreement = gap_by_judge.corr(method="spearman")
        agreement.to_csv(out_dir / f"situation_agreement{suffix}.csv")
        print(f"\nPer-situation {contrast} gap: Spearman correlation between judges\n", agreement.round(3))
        show = (["rank", "gap_mean_over_judges", "n_judges_same_sign"] + [f"gap {j}" for j in judges]
                + ["n_references"] + (["expected_level"] if expected is not None else []) + ["situation"])
        with pd.option_context("display.max_colwidth", 90, "display.float_format", "{:.3f}".format):
            print(f"\nTop {args.top} situations by |{contrast}| (mean over judges)\n",
                  top[show].head(args.top).to_string(index=False))

        # Topics: the same measures by topic domain and subdomain (after the rest, so that adding them
        # does not change the bootstrap draws of the pooled results).
        groupings = available_groupings(df)
        if groupings:
            topics, tests = [], []
            for judge in judges:
                r, t = analyze_topics(df[df["judge"] == judge], variable, groupings, args.n_boot, args.n_perm,
                                      rng, rel)
                topics += r
                tests += t
            topics, tests = pd.DataFrame(topics), pd.DataFrame(tests)
            topics.to_csv(out_dir / f"topics{suffix}.csv", index=False)
            tests.to_csv(out_dir / f"topic_heterogeneity{suffix}.csv", index=False)
            for grouping in groupings:
                plot_topics(topics, gaps, judges, grouping, contrast, args.min_situations, out_dir,
                            f"Signed score gaps by topic {grouping} — {title}", rel, baseline)
            print(f"\nDo the topics differ? eta2 of the situation {'percentages' if rel else 'means'} between "
                  "topic groups vs its value under no topic effect, permutation p-value\n",
                  tests.assign(eta2=tests["eta2"].round(3), eta2_null=tests["eta2_null"].round(3))
                  .pivot(index=["grouping", "measure"], columns="judge", values=["eta2", "p_value"])
                  .to_string())
            print(f"(eta2 under no topic effect: "
                  f"{', '.join(f'{g} {v:.3f}' for g, v in tests.groupby('grouping')['eta2_null'].first().items())})")
            first = next(iter(groupings))
            for measure in [contrast]:
                sub = topics[(topics["grouping"] == first) & (topics["measure"] == measure)]
                print(f"\n{measure} by {first} (mean [95% CI])\n",
                      sub.assign(value=sub.apply(fmt, axis=1))
                      .pivot(index="topic", columns="judge", values="value").to_string())
        else:
            print("\nNo topic columns in the scores: run topics.py to add them.")

        # Stances: the same by the stance of the reference, the pair of stances and their match.
        if all(c in df for c in STANCE_COLUMNS):
            stances, tests = [], []
            for judge in judges:
                r, t = analyze_stances(df[df["judge"] == judge], variable, args.n_boot, args.n_perm, rng, rel)
                stances += r
                tests += t
            stances = pd.DataFrame(stances)
            stances.to_csv(out_dir / f"stances{suffix}.csv", index=False)
            for grouping, (_, order) in STANCE_GROUPINGS.items():
                plot_topics(stances, gaps, judges, grouping, contrast, 0, out_dir,
                            f"Signed score gaps by stance {grouping} — {title}", rel, baseline,
                            order=order, name=f"stances_{grouping}", unit="group")
            if tests:
                tests = pd.DataFrame(tests)
                tests.to_csv(out_dir / "stance_heterogeneity.csv", index=False)
                print("\nDo the stance groups differ? eta2 of the item means between the groups vs its mean under "
                      "stances shuffled within the situations, permutation p-value\n",
                      tests.assign(eta2=tests["eta2"].round(3), eta2_null_mean=tests["eta2_null_mean"].round(3))
                      .pivot(index=["grouping", "measure"], columns="judge",
                             values=["eta2", "eta2_null_mean", "p_value"]).to_string())
            for grouping in STANCE_GROUPINGS:
                for measure in [contrast] + ([CONGRUENT] if CONGRUENT in set(stances["measure"]) else []):
                    sub = stances[(stances["grouping"] == grouping) & (stances["measure"] == measure)]
                    print(f"\n{measure} by stance {grouping} (mean [95% CI])\n",
                          sub.assign(value=sub.apply(fmt, axis=1))
                          .pivot(index="topic", columns="judge", values="value")
                          .reindex(STANCE_GROUPINGS[grouping][1]).dropna(how="all").to_string())

            # Each judge's top situations (ranked as in model_specific_top_situations), with the gap of every
            # stance group (match, pair) of each situation in one plot.
            by_stance = [stance_situations(df[df["judge"] == judge], variable, rel) for judge in judges]
            by_stance = pd.concat([s for s in by_stance if s is not None], ignore_index=True)
            stance_dir = out_dir / "model_specific_top_situations_by_stance"
            stance_dir.mkdir(exist_ok=True)
            for judge in judges:
                ranked = top_situations(by_situation, texts, judges, contrast, expected, rank_by=judge)
                for grouping in ("match", "pair"):
                    sub = by_stance[(by_stance["judge"] == judge) & (by_stance["grouping"] == grouping)
                                    & (by_stance["measure"] == contrast)]
                    name = f"top_situations_{judge}_by_{grouping}"
                    table = stance_top_table(ranked, sub, judge, STANCE_GROUPINGS[grouping][1])
                    table.to_csv(stance_dir / f"{name}{suffix}.csv", index=False)
                    group_means = (stances[(stances["judge"] == judge) & (stances["grouping"] == grouping)
                                           & (stances["measure"] == contrast)].set_index("topic")["mean"].to_dict())
                    plot_stance_top_situations(table, judge, grouping, contrast, args.top, stance_dir,
                                               f"Situations with the largest {contrast} gap for {judge}, by stance "
                                               f"{grouping} — {title}", rel, baseline, name, group_means)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
