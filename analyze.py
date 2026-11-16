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

Usage:
    python analyze.py scores/reference_gender
    python analyze.py scores/reference_gender --out_dir analysis/pilot --n_boot 2000
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


def cluster_bootstrap(values, clusters, n_boot, rng):
    """Mean of `values` and the 95% CI of a bootstrap that resamples whole clusters."""
    ok = ~np.isnan(values)
    values, clusters = values[ok], np.asarray(clusters)[ok]
    if len(values) == 0:
        return np.nan, np.nan, np.nan, 0
    codes, uniques = pd.factorize(clusters)
    sums = np.bincount(codes, weights=values)
    counts = np.bincount(codes).astype(float)
    weights = rng.multinomial(len(uniques), np.full(len(uniques), 1 / len(uniques)), size=n_boot)
    boot = (weights @ sums) / (weights @ counts)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return values.mean(), lo, hi, len(values)


def summarize(values, clusters, n_boot, rng, **keys):
    mean, lo, hi, n = cluster_bootstrap(np.asarray(values, dtype=float), clusters, n_boot, rng)
    return {**keys, "mean": mean, "ci_low": lo, "ci_high": hi, "n": n}


def analyze_judge(df, variable, n_boot, rng):
    judge = df["judge"].iloc[0]
    variation, gaps = [], []

    # Prompt: across the prompt variations, level fixed.
    by_prompt = df.pivot_table(index=["situation_id", "item", "level"], columns="prompt_variation",
                               values="score", aggfunc="first")
    prompt = pairwise(by_prompt)
    situations = prompt.index.get_level_values("situation_id")
    for metric in ("mad", "changed"):
        variation.append(summarize(prompt[metric], situations, n_boot, rng,
                                   judge=judge, factor="prompt", metric=metric))

    levels = ordered_levels(df["level"].unique(), variable)
    if df["level"].nunique() < 2:
        return variation, gaps, None

    by_level = df.pivot_table(index=["situation_id", "reference_original_text_id", "item", "prompt_variation"],
                              columns="level", values="score", aggfunc="first")
    situations = by_level.index.get_level_values("situation_id")
    prompts = by_level.index.get_level_values("prompt_variation")

    # Per observation (item x prompt), for the per-situation table: the signed gaps and the unsigned measures.
    observed = {}

    # Levels: across the non-neutral levels (female vs male), prompt fixed.
    if len(levels) >= 2:
        between = pairwise(by_level[levels])
        for metric in ("mad", "changed"):
            variation.append(summarize(between[metric], situations, n_boot, rng,
                                       judge=judge, factor="levels", metric=metric))
            observed[f"levels {metric}"] = between[metric]

    # Cue: each non-neutral level vs neutral, averaged over the levels.
    has_neutral = NEUTRAL_LEVEL in by_level.columns
    if has_neutral and levels:
        diffs = by_level[levels].sub(by_level[NEUTRAL_LEVEL], axis=0)
        cue = {"mad": diffs.abs().mean(axis=1), "changed": (diffs != 0).where(diffs.notna()).mean(axis=1)}
        for metric, values in cue.items():
            variation.append(summarize(values, situations, n_boot, rng,
                                       judge=judge, factor="cue", metric=metric))
            observed[f"cue {metric}"] = values

    # Signed gaps, pooled and by prompt variation.
    contrasts = {}
    if has_neutral:
        contrasts.update({f"{lv} - {NEUTRAL_LEVEL}": by_level[lv] - by_level[NEUTRAL_LEVEL] for lv in levels})
    contrasts.update({f"{a} - {b}": by_level[a] - by_level[b] for a, b in combinations(levels, 2)})
    for name, gap in contrasts.items():
        gaps.append(summarize(gap, situations, n_boot, rng, judge=judge, contrast=name, prompt_variation="all"))
        for pv in SCORING_PROMPTS:
            mask = np.asarray(prompts == pv)
            if mask.any():
                gaps.append(summarize(gap[mask], situations[mask], n_boot, rng,
                                      judge=judge, contrast=name, prompt_variation=pv))
    by_situation = per_situation(pd.DataFrame({**contrasts, **observed}), judge)
    return variation, gaps, by_situation


def per_situation(observed, judge):
    """Each measure averaged over the observations (item x prompt) of each situation, with its standard
    error clustered by reference opinion (each reference has the same number of candidates and prompts,
    so the mean of the reference means is the mean of the observations). Long format."""
    by_reference = observed.groupby(level=["situation_id", "reference_original_text_id"]).mean()
    grouped = by_reference.groupby(level="situation_id")
    mean, n = grouped.mean(), grouped.size()
    se = grouped.std().div(np.sqrt(n), axis=0)
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


def save(fig, out_dir, name):
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


def plot_variation(variation, factor_labels, judges, out_dir, title):
    metrics = [("mad", "Mean absolute score difference"), ("changed", "Share of score pairs that differ")]
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
    legend(fig, judges, judges)
    fig.suptitle(title, fontsize=11, color=INK, y=1.09)
    fig.tight_layout()
    save(fig, out_dir, "variation")


def plot_gaps(gaps, judges, out_dir, title):
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
    ax.set_xlabel("Mean score difference (95% CI, bootstrap over situations)", fontsize=9, color=INK_2)
    legend(fig, judges, judges)
    fig.suptitle(title, fontsize=11, color=INK, y=1.1)
    fig.tight_layout()
    save(fig, out_dir, "signed_gaps")


def plot_gaps_by_prompt(gaps, judges, out_dir, title):
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
    axes[0].set_ylabel("Mean score difference (95% CI)", fontsize=9, color=INK_2)
    legend(fig, judges, judges)
    fig.suptitle(title, fontsize=11, color=INK, y=1.1)
    fig.tight_layout()
    save(fig, out_dir, "signed_gaps_by_prompt")


def plot_top_situations(top, judges, contrast, n_top, out_dir, title):
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
    ax.set_xlabel(f"Mean {contrast} score difference in the situation\n(±1.96 SE over its references; "
                  f"n = references)", fontsize=9, color=INK_2)
    # Fixed room (in inches) at the top for the title and the legend, whatever the figure's height.
    height = fig.get_figheight()
    fig.suptitle(title, fontsize=11, color=INK, y=1 - 0.15 / height, va="top")
    legend(fig, judges, judges, y=1 - 0.45 / height)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / height))
    save(fig, out_dir, "top_situations")


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
        v, g, s = analyze_judge(df[df["judge"] == judge], variable, args.n_boot, rng)
        variation += v
        gaps += g
        if s is not None:
            by_situation.append(s)
    variation, gaps = pd.DataFrame(variation), pd.DataFrame(gaps)
    mean_scores = (df.groupby(["judge", "level", "prompt_variation"])["score"].mean().unstack()
                   .reindex(columns=[p for p in SCORING_PROMPTS if p in set(df["prompt_variation"])]))
    mean_scores["all"] = df.groupby(["judge", "level"])["score"].mean()

    variation.to_csv(out_dir / "variation.csv", index=False)
    gaps.to_csv(out_dir / "signed_gaps.csv", index=False)
    mean_scores.to_csv(out_dir / "mean_scores.csv")

    factor_labels = {"prompt": "Prompt variation\n(5 prompts, level fixed)",
                     "levels": f"{variable or 'Level'}: {' vs '.join(levels)}\n(prompt fixed)",
                     "cue": f"{variable or 'Level'} cue vs {NEUTRAL_LEVEL}\n(prompt fixed)"}
    title = f"{args.results_dir.name}: {intervened} intervened"
    plot_variation(variation, factor_labels, judges, out_dir, f"Score variation by factor — {title}")
    if len(gaps):
        plot_gaps(gaps, judges, out_dir, f"Signed score gaps — {title}")
        plot_gaps_by_prompt(gaps, judges, out_dir, f"Signed score gaps by prompt variation — {title}")

    pd.set_option("display.width", 250, "display.max_columns", 20)
    fmt = lambda r: f"{r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"  # noqa: E731
    table = variation.assign(value=variation.apply(fmt, axis=1)).pivot(
        index=["judge", "metric"], columns="factor", values="value")
    print("\nUnsigned variation (mean [95% CI])\n", table.reindex(columns=[f for f in factor_labels if f in table]))
    if len(gaps):
        pooled = gaps[gaps["prompt_variation"] == "all"]
        print("\nSigned gaps, pooled over prompts (mean [95% CI])\n",
              pooled.assign(value=pooled.apply(fmt, axis=1)).pivot(index="judge", columns="contrast", values="value"))
    print("\nMean score by level\n", mean_scores["all"].unstack().round(4))

    # Situations: where the gap between the levels is largest.
    if by_situation:
        by_situation = pd.concat(by_situation, ignore_index=True)
        contrast = f"{levels[0]} - {levels[1]}" if len(levels) >= 2 else gaps["contrast"].iloc[0]
        top = top_situations(by_situation, load_situation_texts(args.results_dir), judges, contrast)
        by_situation.to_csv(out_dir / "situations.csv", index=False)
        top.to_csv(out_dir / "top_situations.csv", index=False)
        plot_top_situations(top, judges, contrast, args.top, out_dir,
                            f"Situations with the largest {contrast} gap (mean over judges) — {title}")

        gap_by_judge = (by_situation[by_situation["measure"] == contrast]
                        .pivot(index="situation_id", columns="judge", values="mean").reindex(columns=judges))
        agreement = gap_by_judge.corr(method="spearman")
        agreement.to_csv(out_dir / "situation_agreement.csv")
        print(f"\nPer-situation {contrast} gap: Spearman correlation between judges\n", agreement.round(3))
        show = ["rank", "gap_mean_over_judges", "n_judges_same_sign"] + [f"gap {j}" for j in judges] + [
            "n_references", "situation"]
        with pd.option_context("display.max_colwidth", 90, "display.float_format", "{:.3f}".format):
            print(f"\nTop {args.top} situations by |{contrast}| (mean over judges)\n",
                  top[show].head(args.top).to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
