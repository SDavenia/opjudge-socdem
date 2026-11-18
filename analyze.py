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
  topics.csv                 judge x grouping (topic domain, subdomain) x topic x measure (each signed gap,
                             levels/cue mad): mean and CI over the topic's situations, n_situations
  topic_heterogeneity.csv    judge x grouping x measure: do the topics differ more than random groups of
                             situations? eta2 (between-topic share of the variance of the situation values),
                             its value under no topic effect, (G - 1) / (N - 1), and a permutation p-value
  topics_{domain,subdomain}.png/.pdf  the signed gaps by topic (groups of >= --min_situations situations)
The topics need the topic columns of topics.py in the scores.

Usage:
    python analyze.py scores/reference_gender
    python analyze.py scores/reference_gender --out_dir analysis/pilot --n_boot 2000
    python analyze.py scores/reference_gender --relative     # -> *_relative.{csv,png,pdf}
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

# Judges keep their color and marker whatever subset is plotted: slots in the order of
# judges.REGISTRY (then OpenRouter judges, alphabetically). Categorical palette of the dataviz
# reference (light mode); the marker is the secondary encoding past 3 judges.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
INK, INK_2, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, help="A directory of score.py, e.g. scores/reference_gender.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Default: analysis/{results_dir name}.")
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

def load(results_dir, judges=None):
    frames = []
    for path in sorted(results_dir.glob("*_by_prompt.csv")):
        judge = path.name.removesuffix("_by_prompt.csv")
        if judges and judge not in judges:
            continue
        df = pd.read_csv(path)
        df["judge"] = judge
        frames.append(df)
    if not frames:
        raise SystemExit(f"No *_by_prompt.csv in {results_dir}")
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


def cluster_bootstrap(values, clusters, n_boot, rng, denominators=None):
    """Mean of `values` and the 95% CI of a bootstrap that resamples whole clusters.

    With `denominators`, a relative mean: each cluster's mean value divided by its mean denominator (e.g.
    a situation's mean gap / its mean neutral score), then the unweighted mean of those cluster ratios.
    Clusters whose mean denominator is 0 are dropped."""
    values = np.asarray(values, dtype=float)
    relative = denominators is not None
    denominators = np.asarray(denominators, dtype=float) if relative else np.ones_like(values)
    ok = ~np.isnan(values) & ~np.isnan(denominators)
    values, denominators, clusters = values[ok], denominators[ok], np.asarray(clusters)[ok]
    if len(values) == 0:
        return np.nan, np.nan, np.nan, 0
    codes, uniques = pd.factorize(clusters)
    sums = np.bincount(codes, weights=values, minlength=len(uniques))
    den_sums = np.bincount(codes, weights=denominators, minlength=len(uniques))
    if relative:
        ratios = sums[den_sums != 0] / den_sums[den_sums != 0]
        weights = rng.multinomial(len(ratios), np.full(len(ratios), 1 / len(ratios)), size=n_boot)
        boot = weights @ ratios / len(ratios)
        mean = ratios.mean()
    else:
        weights = rng.multinomial(len(uniques), np.full(len(uniques), 1 / len(uniques)), size=n_boot)
        boot = (weights @ sums) / (weights @ den_sums)
        mean = values.mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, lo, hi, len(values)


def summarize(values, clusters, n_boot, rng, denominators=None, **keys):
    mean, lo, hi, n = cluster_bootstrap(values, clusters, n_boot, rng, denominators)
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

    signed, unsigned = level_measures(by_level, levels, relative)

    # Unsigned: levels (female vs male) and cue (each level vs neutral), prompt fixed.
    for name, (values, base) in unsigned.items():
        factor, metric = name.split(" ")
        variation.append(summarize(values, situations, n_boot, rng, base, judge=judge, factor=factor, metric=metric))

    # Signed gaps, pooled and by prompt variation.
    for name, (gap, base) in signed.items():
        gaps.append(summarize(gap, situations, n_boot, rng, base,
                              judge=judge, contrast=name, prompt_variation="all"))
        for pv in SCORING_PROMPTS:
            mask = np.asarray(prompts == pv)
            if mask.any():
                gaps.append(summarize(gap[mask], situations[mask], n_boot, rng,
                                      None if base is None else base[mask],
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


def level_measures(by_level, levels, relative):
    """The per-observation measures of the levels, as (signed, unsigned): dicts measure -> (values, baseline),
    the baseline being None in absolute mode or for `changed` (already scale-free).

    signed    each level - neutral and each pair of levels (female - male); baseline: the neutral score
              (or, with no neutral level, the subtracted level's)
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
    return signed, unsigned


# ---------------------------------------------------------------------------------------------
# Topics

TOPIC_GROUPINGS = {"domain": "topic_domain", "subdomain": "topic_subdomain"}


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


def analyze_topics(df, variable, n_boot, n_perm, rng, relative=False):
    """Every signed gap and unsigned mad by topic domain and subdomain (mean and bootstrap CI over the
    situations of the group, absolute or relative as in the pooled analysis), and a permutation test of
    whether they differ between the groups."""
    judge = df["judge"].iloc[0]
    levels = ordered_levels(df["level"].unique(), variable)
    by_level = level_table(df)
    situations = by_level.index.get_level_values("situation_id")
    signed, unsigned = level_measures(by_level, levels, relative)
    measures = {**signed, **{m: v for m, v in unsigned.items() if m.endswith("mad")}}
    topic_of = df.drop_duplicates("situation_id").set_index("situation_id")

    rows, tests = [], []
    for grouping, column in TOPIC_GROUPINGS.items():
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


def top_situations(by_situation, situations_text, judges, contrast):
    """One row per situation: the gap of `contrast` for each judge, its mean over the judges, and the
    mean unsigned difference between the levels; sorted by the absolute mean gap over the judges."""
    gap = by_situation[by_situation["measure"] == contrast]
    wide = gap.pivot(index="situation_id", columns="judge", values="mean").reindex(columns=judges)
    se = gap.pivot(index="situation_id", columns="judge", values="se").reindex(columns=judges)
    out = pd.DataFrame(index=wide.index)
    out["situation"] = situations_text.reindex(wide.index)
    out["n_references"] = gap.groupby("situation_id")["n_references"].first()
    out["gap_mean_over_judges"] = wide.mean(axis=1)
    out["n_judges_same_sign"] = (np.sign(wide).eq(np.sign(out["gap_mean_over_judges"]), axis=0)).sum(axis=1)
    for judge in judges:
        out[f"gap {judge}"] = wide[judge]
        out[f"se {judge}"] = se[judge]
    unsigned = by_situation[by_situation["measure"] == "levels mad"]
    if len(unsigned):
        out["levels_mad_mean_over_judges"] = unsigned.groupby("situation_id")["mean"].mean()
    out = out.reindex(out["gap_mean_over_judges"].abs().sort_values(ascending=False).index)
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


def plot_gaps(gaps, judges, out_dir, title, relative=False, baseline=NEUTRAL_LEVEL):
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
    save(fig, out_dir, "signed_gaps", relative)


def plot_gaps_by_prompt(gaps, judges, out_dir, title, relative=False, baseline=NEUTRAL_LEVEL):
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
    save(fig, out_dir, "signed_gaps_by_prompt", relative)


def plot_top_situations(top, judges, contrast, n_top, out_dir, title, relative=False, baseline=NEUTRAL_LEVEL):
    rows = top.head(n_top)
    fig, axes = figure(figsize=(10, 1.0 + 0.55 * len(rows) * max(1, len(judges) / 3)))
    ax = axes[0]
    setup_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=1)
    step = 0.6 / max(1, len(judges) - 1) if len(judges) > 1 else 0
    for k, judge in enumerate(judges):
        color, marker = style_of(judge, judges)
        y = np.arange(len(rows)) + (k - (len(judges) - 1) / 2) * step
        ax.errorbar(rows[f"gap {judge}"], y, xerr=1.96 * rows[f"se {judge}"].fillna(0), fmt="none", ecolor=color,
                    elinewidth=2, capsize=0, zorder=2)
        ax.scatter(rows[f"gap {judge}"], y, color=color, marker=marker, s=40, edgecolors=SURFACE, linewidths=1.5,
                   zorder=3)
    labels = [f"{r['rank']}. {textwrap.shorten(str(r['situation']), 70)}  (n={r['n_references']})"
              for _, r in rows.iterrows()]
    ax.set_yticks(np.arange(len(rows)), labels, fontsize=8, color=INK)
    ax.invert_yaxis()
    what = (f"Mean {contrast} score difference in the situation, % of its mean {baseline} score" if relative
            else f"Mean {contrast} score difference in the situation")
    ax.set_xlabel(f"{what}\n(±1.96 SE over its references; n = references)", fontsize=9, color=INK_2)
    if relative:
        percent_axis(ax.xaxis)
    # Fixed room (in inches) at the top for the title and the legend, whatever the figure's height.
    height = fig.get_figheight()
    fig.suptitle(title, fontsize=11, color=INK, y=1 - 0.15 / height, va="top")
    legend(fig, judges, judges, y=1 - 0.45 / height)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / height))
    save(fig, out_dir, "top_situations", relative)


def plot_topics(topics, gaps, judges, grouping, sort_by, min_situations, out_dir, title, relative=False,
                baseline=NEUTRAL_LEVEL):
    """One panel per signed contrast, one row per topic group (with at least `min_situations` situations),
    sorted by the `sort_by` gap averaged over the judges; dotted lines: each judge's pooled gap."""
    pooled = gaps[gaps["prompt_variation"] == "all"]
    contrasts = list(dict.fromkeys(pooled["contrast"]))
    sub = topics[(topics["grouping"] == grouping) & (topics["n_situations"] >= min_situations)]
    order = (sub[sub["measure"] == sort_by].groupby("topic")["mean"].mean().sort_values(ascending=False).index)
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
    fig.supxlabel(f"{gap_label(relative, baseline)} (95% CI, bootstrap over the topic's situations; "
                  f"dotted: pooled; n = situations)", fontsize=9, color=INK_2)
    height = fig.get_figheight()
    fig.suptitle(title, fontsize=11, color=INK, y=1 - 0.15 / height, va="top")
    legend(fig, judges, judges, y=1 - 0.45 / height)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / height))
    save(fig, out_dir, f"topics_{grouping}", relative)


def load_situation_texts(results_dir):
    frames = [pd.read_csv(p, usecols=["situation_id", "situation"]) for p in results_dir.glob("*_mean.csv")]
    if not frames:
        return pd.Series(dtype=str)
    return pd.concat(frames).drop_duplicates("situation_id").set_index("situation_id")["situation"]


# ---------------------------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = args.out_dir or ANALYSIS_DIR / args.results_dir.resolve().name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df, variable, sides = load(args.results_dir, args.judges)
    judges = judge_order(df["judge"].unique())
    levels = ordered_levels(df["level"].unique(), variable)
    intervened = " & ".join(sides) or "no side"
    print(f"{args.results_dir}: judges {judges}, variable {variable}, intervened {intervened}, levels "
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
    title = f"{args.results_dir.name}: {intervened} intervened" + (" (relative)" if rel else "")
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
        top = top_situations(by_situation, load_situation_texts(args.results_dir), judges, contrast)
        by_situation.to_csv(out_dir / f"situations{suffix}.csv", index=False)
        top.to_csv(out_dir / f"top_situations{suffix}.csv", index=False)
        plot_top_situations(top, judges, contrast, args.top, out_dir,
                            f"Situations with the largest {contrast} gap (mean over judges) — {title}", rel, baseline)

        gap_by_judge = (by_situation[by_situation["measure"] == contrast]
                        .pivot(index="situation_id", columns="judge", values="mean").reindex(columns=judges))
        agreement = gap_by_judge.corr(method="spearman")
        agreement.to_csv(out_dir / f"situation_agreement{suffix}.csv")
        print(f"\nPer-situation {contrast} gap: Spearman correlation between judges\n", agreement.round(3))
        show = ["rank", "gap_mean_over_judges", "n_judges_same_sign"] + [f"gap {j}" for j in judges] + [
            "n_references", "situation"]
        with pd.option_context("display.max_colwidth", 90, "display.float_format", "{:.3f}".format):
            print(f"\nTop {args.top} situations by |{contrast}| (mean over judges)\n",
                  top[show].head(args.top).to_string(index=False))

        # Topics: the same measures by topic domain and subdomain (after the rest, so that adding them
        # does not change the bootstrap draws of the pooled results).
        if all(c in df for c in TOPIC_GROUPINGS.values()):
            topics, tests = [], []
            for judge in judges:
                r, t = analyze_topics(df[df["judge"] == judge], variable, args.n_boot, args.n_perm, rng, rel)
                topics += r
                tests += t
            topics, tests = pd.DataFrame(topics), pd.DataFrame(tests)
            topics.to_csv(out_dir / f"topics{suffix}.csv", index=False)
            tests.to_csv(out_dir / f"topic_heterogeneity{suffix}.csv", index=False)
            for grouping in TOPIC_GROUPINGS:
                plot_topics(topics, gaps, judges, grouping, contrast, args.min_situations, out_dir,
                            f"Signed score gaps by topic {grouping} — {title}", rel, baseline)
            print(f"\nDo the topics differ? eta2 of the situation {'percentages' if rel else 'means'} between "
                  "topic groups vs its value under no topic effect, permutation p-value\n",
                  tests.assign(eta2=tests["eta2"].round(3), eta2_null=tests["eta2_null"].round(3))
                  .pivot(index=["grouping", "measure"], columns="judge", values=["eta2", "p_value"])
                  .to_string())
            print(f"(eta2 under no topic effect: "
                  f"{', '.join(f'{g} {v:.3f}' for g, v in tests.groupby('grouping')['eta2_null'].first().items())})")
            domain = topics[(topics["grouping"] == "domain") & (topics["measure"] == contrast)]
            print(f"\n{contrast} by topic domain (mean [95% CI])\n",
                  domain.assign(value=domain.apply(fmt, axis=1))
                  .pivot(index="topic", columns="judge", values="value").to_string())
        else:
            print("\nNo topic columns in the scores: run topics.py to add them.")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
