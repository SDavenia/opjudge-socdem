"""
Analysis of the double intervention (setup "both", e.g. scores/hb_both_gender): the sociodemographic cue is on
the reference AND on the candidate, in every combination of their levels (3 x 3 cells for gender: F, M, N =
female, male, neutral "a person"). Cell "F|M" = reference female, candidate male. Baseline: N|N.

Same-level cells share the identity phrase ("As a woman" on both sides), so they are expected to score higher
by lexical overlap alone. N|N ("As a person" on both sides) has the same overlap with no sociodemographic in
it, and is the control for it.

Per-observation measures (observation = reference x candidate x prompt variation; for gender):
  main         ref F - N, ref M - N, ref F - M      (candidate N)   = F|N - N|N, M|N - N|N, F|N - M|N
               cand F - N, cand M - N               (reference N)   = N|F - N|N, N|M - N|N
  interaction  int X|Y = X|Y - X|N - N|Y + N|N, for X, Y in {F, M}: the non-additive part of a cell
  matching     same - cross     (F|F + M|M - F|M - M|F) / 2: identical gendered phrase vs one word different
               N|N - mixed      N|N - (F|N + M|N + N|F + N|M) / 4: identical neutral phrase vs one word
                                different (the lexical-overlap control)
               match excess     (same - cross) - (N|N - mixed): the gender match beyond phrase identity
  asymmetry    F|F - M|M (which match scores higher), F|M - M|F (which cross)
  softening    cand F - M | ref N, F, M  (= N|F - N|M, F|F - F|M, M|F - M|M): the candidate female - male gap
               (the "softened disagreement" of the candidate setup) by the reference level
  congruence   ref cong, cand cong, match cong: the expected level of the situation (expected_authority,
               M -> male, W -> female) minus the other one, on the reference (cand N), on the candidate
               (ref N), on both (E|E - O|O); pooled with the M and W situations weighted equally (stratified
               bootstrap), so a gap shared by all the situations cancels out
--relative: each measure over the N|N score (per situation, as in analyze.py).

Breakdowns (95% CIs by bootstrap over situations, as in analyze.py): pooled and by prompt variation; by
situation (SE clustered by reference, Spearman agreement between judges); by situation grouping
(selection_topic, expected_authority, topic domain/subdomain, with the permutation test of analyze.py); by
stance (reference stance, stance pair, stance match, with the within-situation permutation test).

Decision flips: do the cues change which candidate the judge prefers? For a reference with gender X (O = the
other gender) and two candidates a, b of its situation with a scored strictly higher than b when both
candidates are N (same prompt), b "overtakes" a when its score is strictly higher after:
  match vs cross    b gets X (matches the reference), a gets O
  cross vs match    the reverse assignment (control: noise and any gender effect not tied to matching)
  match vs neutral  b gets X, a stays N
  phrase control    reference N, both candidates G (G in {F, M}) as the baseline; b gets N (identical phrase
                    "a person"), a stays G: the lexical analogue of "match vs neutral"
by pair (a over b, rate over the eligible pairs; also over the close ones, baseline margin <= CLOSE = 0.1) and by ranking (top-1: the top candidate under the baseline
loses its place when every other candidate gets the favourable cue), pooled and by the stance relation of a
and b to the reference (same / partial / opposite), e.g. "same over opposite" = an aligned candidate ahead of
an opposed one.

Replication: the main effects next to the single-side setups (analysis/{reference,candidate}_gender), whose
other side is the original opinion with no preamble, not N.

Output in {out_dir} (default analysis/{name of results_dir without the dataset prefix}, e.g. both_gender):
  cells.csv                        judge x cell: mean score; cells.png: difference from N|N
  measures.csv                     judge x measure x prompt variation ("all" = pooled): mean, CI
  measures_{family}.png            the pooled measures of a family, by judge
  measures_by_prompt_{family}.png  the same by prompt variation (matching, softening)
  situations.csv, situation_agreement.csv   per-situation measures, judge agreement on the key measures
  groups.csv, group_heterogeneity.csv, groups_{grouping}_{family}.png       by situation grouping
  stances.csv, stance_heterogeneity.csv, stances_{grouping}_{family}.png    by stance
  flips.csv, flips.png             the decision flips
  replication.csv                  main effects vs the single-side setups

Usage:
    python analyze_both.py scores/hb_both_gender --out_dir analysis/both_gender
    python analyze_both.py scores/hb_both_gender --out_dir analysis/both_gender --relative
"""
import argparse
from itertools import permutations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

import analyze as A  # noqa: E402
from add_sociodem_preamble import NEUTRAL_LEVEL  # noqa: E402
from prompts import SCORING_PROMPTS  # noqa: E402

N = NEUTRAL_LEVEL
SHORT = {"female": "F", "male": "M", NEUTRAL_LEVEL: "N"}
FAMILIES = {
    "main": ["ref F - N", "ref M - N", "ref F - M", "cand F - N", "cand M - N"],
    "interaction": ["int F|F", "int M|M", "int F|M", "int M|F"],
    "matching": ["same - cross", "N|N - mixed", "match excess"],
    "asymmetry": ["F|F - M|M", "F|M - M|F"],
    "softening": ["cand F - M | ref N", "cand F - M | ref F", "cand F - M | ref M"],
    "congruence": ["ref cong", "cand cong", "match cong"],
}
CONGRUENCE = set(FAMILIES["congruence"])
KEY_MEASURES = ["match excess", "same - cross", "cand F - M | ref N", "F|F - M|M"]
# The single-side contrast that each main effect replicates: (setup directory, contrast).
REPLICATION = {"ref F - N": ("reference_gender", "female - neutral"),
               "ref M - N": ("reference_gender", "male - neutral"),
               "ref F - M": ("reference_gender", "female - male"),
               "cand F - N": ("candidate_gender", "female - neutral"),
               "cand M - N": ("candidate_gender", "male - neutral"),
               "cand F - M | ref N": ("candidate_gender", "female - male"),
               "ref cong": ("reference_gender", A.CONGRUENT), "cand cong": ("candidate_gender", A.CONGRUENT)}
CLOSE = 0.1  # baseline margin of a "close" pair or ranking
DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#2a78d6", "#f0efec", "#e34948"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dirs", type=Path, nargs="+",
                        help="'both' directories of score.py, e.g. scores/hb_both_gender; several are analysed jointly.")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--judges", nargs="+", default=None)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--n_perm", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_situations", type=int, default=5)
    parser.add_argument("--single_side_dir", type=Path, default=None,
                        help="Where the single-side analyses are (for the replication table; default: the "
                             "parent of the output directory, e.g. analysis/vp).")
    parser.add_argument("--relative", action="store_true", help="Measures relative to the N|N score.")
    return parser.parse_args()


# ---------------------------------------------------------------------------------------------
# Measures

def cell(by_level, ref, cand):
    return by_level[f"{ref}|{cand}"]


def both_measures(by_level, relative, expected=None):
    """name -> (per-observation values, baseline or None), for the observations (rows) of by_level."""
    c = lambda r, k: cell(by_level, r, k)  # noqa: E731
    F, M = "female", "male"
    nn = c(N, N)
    out = {
        "ref F - N": c(F, N) - nn, "ref M - N": c(M, N) - nn, "ref F - M": c(F, N) - c(M, N),
        "cand F - N": c(N, F) - nn, "cand M - N": c(N, M) - nn,
    }
    for x in (F, M):
        for y in (F, M):
            out[f"int {SHORT[x]}|{SHORT[y]}"] = c(x, y) - c(x, N) - c(N, y) + nn
    same_cross = (c(F, F) + c(M, M) - c(F, M) - c(M, F)) / 2
    phrase = nn - (c(F, N) + c(M, N) + c(N, F) + c(N, M)) / 4
    out.update({"same - cross": same_cross, "N|N - mixed": phrase, "match excess": same_cross - phrase,
                "F|F - M|M": c(F, F) - c(M, M), "F|M - M|F": c(F, M) - c(M, F),
                "cand F - M | ref N": c(N, F) - c(N, M), "cand F - M | ref F": c(F, F) - c(F, M),
                "cand F - M | ref M": c(M, F) - c(M, M)})
    if expected is not None:
        exp = expected.reindex(by_level.index.get_level_values("situation_id")).to_numpy()
        sign = np.where(exp == F, 1.0, np.where(exp == M, -1.0, np.nan))
        out.update({"ref cong": (c(F, N) - c(M, N)) * sign, "cand cong": (c(N, F) - c(N, M)) * sign,
                    "match cong": (c(F, F) - c(M, M)) * sign})
    return {name: (values, nn if relative else None) for name, values in out.items()}


def family_of(measure):
    return next(f for f, ms in FAMILIES.items() if measure in ms)


# ---------------------------------------------------------------------------------------------
# Decision flips

def stance_relation(ref, cand):
    return A.stance_match(ref, cand)


def flips(df, n_boot, rng):
    """The decision flips of one judge (see the docstring): rows of (metric, unit, relation, mean, CI)."""
    judge = df["judge"].iloc[0]
    S = df.pivot_table(index=["situation_id", "reference_original_text_id", "prompt_variation"],
                       columns=["reference_level", "candidate_stance_group", "candidate_level"],
                       values="score", aggfunc="first")
    ref_stance = (df.drop_duplicates("reference_original_text_id")
                  .set_index("reference_original_text_id")["reference_stance_group"]
                  .reindex(S.index.get_level_values("reference_original_text_id")).to_numpy())
    situations = S.index.get_level_values("situation_id").to_numpy()
    stances = [s for s in A.STANCES if s in set(df["candidate_stance_group"])]
    other = {"female": "male", "male": "female"}

    # Each scenario: (reference level of the baseline and of the intervention, baseline candidate level,
    # level of the overtaking candidate b, level of the overtaken candidate a). One per gender, pooled.
    scenarios = {
        "match vs cross": [(g, N, g, other[g]) for g in other],
        "cross vs match": [(g, N, other[g], g) for g in other],
        "match vs neutral": [(g, N, g, N) for g in other],
        "phrase control": [(N, g, N, g) for g in other],
    }
    rows = []
    for metric, specs in scenarios.items():
        pair_vals, pair_rel, pair_sit, pair_close, top_vals, top_sit, top_rel, top_close = ([] for _ in range(8))
        for ref_lv, base_lv, b_lv, a_lv in specs:
            def score(stance, level):
                return S[(ref_lv, stance, level)].to_numpy()
            base = {s: score(s, base_lv) for s in stances}
            for a, b in permutations(stances, 2):
                eligible = base[a] > base[b]
                over = (score(b, b_lv) > score(a, a_lv)).astype(float)
                pair_vals.append(np.where(eligible, over, np.nan))
                pair_sit.append(situations)
                pair_close.append(base[a] - base[b] <= CLOSE)
                pair_rel.append(np.char.add(np.char.add(stance_relation(ref_stance, a).astype(str), " over "),
                                            stance_relation(ref_stance, b).astype(str)))
            # Top-1: the strictly best candidate under the baseline; every other one gets b's level, it a's.
            base_mat = np.stack([base[s] for s in stances], axis=1)
            best = np.nanargmax(np.where(np.isnan(base_mat), -np.inf, base_mat), axis=1)
            unique = (base_mat == base_mat.max(axis=1, keepdims=True)).sum(axis=1) == 1
            top_score = np.stack([score(s, a_lv) for s in stances], axis=1)[np.arange(len(best)), best]
            others = np.stack([score(s, b_lv) for s in stances], axis=1)
            others[np.arange(len(best)), best] = -np.inf
            top_vals.append(np.where(unique, (others.max(axis=1) > top_score).astype(float), np.nan))
            top_sit.append(situations)
            second = np.sort(np.where(np.isnan(base_mat), -np.inf, base_mat), axis=1)[:, -2]
            top_close.append(base_mat.max(axis=1) - second <= CLOSE)
            top_rel.append(stance_relation(ref_stance, np.asarray(stances, dtype=object)[best]))
        for unit, vals, sits, rels, close in (("pair", pair_vals, pair_sit, pair_rel, pair_close),
                                              ("top-1", top_vals, top_sit, top_rel, top_close)):
            vals, sits, rels, close = (np.concatenate(x) for x in (vals, sits, rels, close))
            groups = ([("all", np.ones(len(vals), dtype=bool)), (f"close (margin <= {CLOSE})", close)]
                      + [(r, rels == r) for r in sorted(pd.unique(rels))])
            for relation, mask in groups:
                ok = mask & ~np.isnan(vals)
                if ok.sum() == 0:
                    continue
                rows.append({**A.summarize(vals[ok], sits[ok], n_boot, rng, judge=judge, metric=metric, unit=unit,
                                           relation=relation), "n_eligible": int(ok.sum())})
    return rows


# ---------------------------------------------------------------------------------------------
# Plots

def plot_cells(cells, judges, out_dir, title):
    """One 3 x 3 heatmap per judge: cell mean - N|N mean, on a shared diverging scale."""
    levels = ["female", "male", N]
    fig, axes = A.figure(ncols=len(judges), figsize=(3.3 * len(judges) + 0.8, 3.4))
    diffs = {j: cells[cells["judge"] == j].pivot(index="reference_level", columns="candidate_level",
                                                  values="score").reindex(index=levels, columns=levels)
             for j in judges}
    diffs = {j: d - d.loc[N, N] for j, d in diffs.items()}
    vmax = max(np.nanmax(np.abs(d.to_numpy())) for d in diffs.values())
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    for ax, judge in zip(axes, judges):
        d = diffs[judge].to_numpy()
        image = ax.imshow(d, cmap=DIVERGING, norm=norm)
        for i in range(3):
            for k in range(3):
                ax.text(k, i, f"{d[i, k]:+.3f}", ha="center", va="center", fontsize=9, color=A.INK)
        # 2px surface gaps between the cells.
        for edge in (0.5, 1.5):
            ax.axhline(edge, color=A.SURFACE, linewidth=2)
            ax.axvline(edge, color=A.SURFACE, linewidth=2)
        ax.set_xticks(range(3), [SHORT[lv] for lv in levels], fontsize=9, color=A.INK)
        ax.set_yticks(range(3), [SHORT[lv] for lv in levels], fontsize=9, color=A.INK)
        ax.set_xlabel("candidate", fontsize=9, color=A.INK_2)
        ax.set_title(judge, fontsize=10, color=A.INK)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_ylabel("reference", fontsize=9, color=A.INK_2)
    bar = fig.colorbar(image, ax=list(axes), shrink=0.8, pad=0.02)
    bar.set_label("Mean score - mean N|N score", fontsize=9, color=A.INK_2)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8, colors=A.INK_2, length=0)
    fig.suptitle(title, fontsize=11, color=A.INK, y=1.02)
    A.save(fig, out_dir, "cells")


def plot_flips(flip_rows, judges, out_dir, title):
    """Pair-level overtaking rate by scenario (pooled and 'same over opposite'); one panel per relation."""
    metrics = ["match vs cross", "cross vs match", "match vs neutral", "phrase control"]
    relations = [("all", "All eligible pairs"), (f"close (margin <= {CLOSE})", f"Close pairs (margin <= {CLOSE})"),
                 ("same over opposite", "Aligned ahead of an opposed one"),
                 ("same over partial", "Aligned ahead of a neutral-stance one")]
    sub = flip_rows[flip_rows["unit"] == "pair"]
    fig, axes = A.figure(ncols=len(relations), figsize=(4.4 * len(relations), 3.6), sharey=True)
    step = 0.6 / max(1, len(judges) - 1)
    for ax, (relation, label) in zip(axes, relations):
        A.setup_axes(ax)
        ax.grid(axis="y", visible=False)
        for k, judge in enumerate(judges):
            color, marker = A.style_of(judge, judges)
            rows = sub[(sub["judge"] == judge) & (sub["relation"] == relation)].set_index("metric").reindex(metrics)
            y = np.arange(len(metrics)) + (k - (len(judges) - 1) / 2) * step
            ax.errorbar(rows["mean"], y, xerr=[rows["mean"] - rows["ci_low"], rows["ci_high"] - rows["mean"]],
                        fmt="none", ecolor=color, elinewidth=2, capsize=0, zorder=2)
            ax.scatter(rows["mean"], y, color=color, marker=marker, s=40, edgecolors=A.SURFACE, linewidths=1.5,
                       zorder=3)
        ax.set_title(label, fontsize=10, color=A.INK)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    axes[0].set_yticks(np.arange(len(metrics)), metrics, fontsize=9, color=A.INK)
    axes[0].invert_yaxis()
    fig.supxlabel("Share of pairs where the lower-ranked candidate overtakes (95% CI, bootstrap over situations)",
                  fontsize=9, color=A.INK_2)
    A.legend(fig, judges, judges, y=1.06)
    fig.suptitle(title, fontsize=11, color=A.INK, y=1.14)
    fig.tight_layout()
    A.save(fig, out_dir, "flips")


# ---------------------------------------------------------------------------------------------

def main():
    args = parse_args()
    name = A.run_name(args.results_dirs)
    out_dir = args.out_dir or A.default_out_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.single_side_dir is None:
        args.single_side_dir = out_dir.parent
    rng = np.random.default_rng(args.seed)
    rel = args.relative
    suffix = "_relative" if rel else ""

    df, variable, sides = A.load(args.results_dirs, args.judges)
    if sides != ["reference", "candidate"]:
        raise SystemExit(f"{name} is not a double intervention (intervened: {sides})")
    judges = A.judge_order(df["judge"].unique())
    print(f"{name}: judges {judges}, variable {variable}, cells {sorted(df['level'].unique())}")

    cells = df.groupby(["judge", "reference_level", "candidate_level"])["score"].mean().rename("score").reset_index()
    cells.to_csv(out_dir / "cells.csv", index=False)
    title = f"{name}: both sides intervened" + (" (relative to N|N)" if rel else "")
    if not rel:
        plot_cells(cells, judges, out_dir, f"Mean score by cell, minus N|N — {name}")

    measures_rows, by_situation, group_rows, group_tests, stance_rows, stance_tests, flip_rows = ([] for _ in range(7))
    groupings = {g: c for g, c in A.available_groupings(df).items()}
    for judge in judges:
        jdf = df[df["judge"] == judge]
        by_level = A.level_table(jdf)
        situations = by_level.index.get_level_values("situation_id")
        prompts = by_level.index.get_level_values("prompt_variation")
        expected = A.expected_levels(jdf, variable)
        measures = both_measures(by_level, rel, expected)
        strata = {} if expected is None else {m: expected.reindex(situations).to_numpy() for m in CONGRUENCE}
        for measure, (values, base) in measures.items():
            for pv in ["all"] + list(SCORING_PROMPTS):
                mask = np.ones(len(values), dtype=bool) if pv == "all" else np.asarray(prompts == pv)
                measures_rows.append(A.summarize(
                    values[mask], situations[mask], args.n_boot, rng, None if base is None else base[mask],
                    strata[measure][mask] if measure in strata else None,
                    judge=judge, family=family_of(measure), contrast=measure, prompt_variation=pv))
        by_situation.append(A.per_situation(pd.DataFrame({m: v for m, (v, _) in measures.items()}), judge,
                                            {m: b for m, (_, b) in measures.items() if b is not None}))
        no_cong = {m: v for m, v in measures.items() if m not in CONGRUENCE}
        if groupings:
            r, t = A.topic_breakdown(jdf, by_level, no_cong, groupings, args.n_boot, args.n_perm, rng)
            group_rows += r
            group_tests += t
        r, t = A.stance_breakdown(jdf, by_level, measures, args.n_boot, args.n_perm, rng, rel, strata)
        stance_rows += r
        stance_tests += t
        if not rel:
            flip_rows += flips(jdf, args.n_boot, rng)

    gaps = pd.DataFrame(measures_rows)
    gaps.to_csv(out_dir / f"measures{suffix}.csv", index=False)
    by_situation = pd.concat(by_situation, ignore_index=True)
    by_situation.to_csv(out_dir / f"situations{suffix}.csv", index=False)
    agreement = []
    for measure in KEY_MEASURES:
        wide = (by_situation[by_situation["measure"] == measure]
                .pivot(index="situation_id", columns="judge", values="mean").reindex(columns=judges))
        corr = wide.corr(method="spearman")
        agreement += [{"measure": measure, "judge_a": a, "judge_b": b, "spearman": corr.loc[a, b]}
                      for i, a in enumerate(judges) for b in judges[i + 1:]]
    agreement = pd.DataFrame(agreement)
    agreement.to_csv(out_dir / f"situation_agreement{suffix}.csv", index=False)

    baseline = "N|N"
    for family, members in FAMILIES.items():
        sub = gaps[gaps["contrast"].isin(members)]
        if not len(sub):
            continue
        A.plot_gaps(sub, judges, out_dir, f"{family} — {title}", rel, baseline, name=f"measures_{family}")
        if family in ("matching", "softening", "main"):
            A.plot_gaps_by_prompt(sub, judges, out_dir, f"{family} by prompt variation — {title}", rel, baseline,
                                  name=f"measures_by_prompt_{family}")

    plot_families = ["matching", "softening", "asymmetry", "interaction"]
    if group_rows:
        groups, tests = pd.DataFrame(group_rows), pd.DataFrame(group_tests)
        groups.to_csv(out_dir / f"groups{suffix}.csv", index=False)
        tests.to_csv(out_dir / f"group_heterogeneity{suffix}.csv", index=False)
        for grouping in [g for g in ("selection", "authority") if g in groupings]:
            for family in plot_families[:3]:
                A.plot_topics(groups, gaps[gaps["contrast"].isin(FAMILIES[family])], judges, grouping,
                              FAMILIES[family][0], args.min_situations, out_dir,
                              f"{family} by {grouping} — {title}", rel, baseline,
                              name=f"groups_{grouping}_{family}")
    stances = pd.DataFrame(stance_rows)
    stances.to_csv(out_dir / f"stances{suffix}.csv", index=False)
    for grouping, (_, order) in A.STANCE_GROUPINGS.items():
        for family in plot_families + ["congruence"]:
            A.plot_topics(stances, gaps[gaps["contrast"].isin(FAMILIES[family])], judges, grouping,
                          FAMILIES[family][0], 0, out_dir, f"{family} by stance {grouping} — {title}", rel,
                          baseline, order=order, name=f"stances_{grouping}_{family}", unit="group")
    if stance_tests:
        pd.DataFrame(stance_tests).to_csv(out_dir / "stance_heterogeneity.csv", index=False)

    # Replication: the main effects vs the single-side setups.
    replication = []
    pooled = gaps[gaps["prompt_variation"] == "all"]
    for measure, (setup, contrast) in REPLICATION.items():
        path = args.single_side_dir / setup / f"signed_gaps{suffix}.csv"
        if not path.exists():
            continue
        single = pd.read_csv(path)
        single = single[(single["prompt_variation"] == "all") & (single["contrast"] == contrast)].set_index("judge")
        for judge in judges:
            mine = pooled[(pooled["judge"] == judge) & (pooled["contrast"] == measure)].iloc[0]
            if judge in single.index:
                s = single.loc[judge]
                replication.append({"judge": judge, "measure": measure, "single_side": f"{setup}: {contrast}",
                                    "both": mine["mean"], "both_ci_low": mine["ci_low"], "both_ci_high": mine["ci_high"],
                                    "single": s["mean"], "single_ci_low": s["ci_low"], "single_ci_high": s["ci_high"]})
    replication = pd.DataFrame(replication)
    replication.to_csv(out_dir / f"replication{suffix}.csv", index=False)

    # Console summary.
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    fmt = ((lambda r: f"{r['mean']:+.2%} [{r['ci_low']:+.2%}, {r['ci_high']:+.2%}]") if rel
           else (lambda r: f"{r['mean']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"))
    print("\nMean score by cell (rows: reference, columns: candidate)")
    for judge in judges:
        print(judge, "\n", cells[cells["judge"] == judge].pivot(index="reference_level", columns="candidate_level",
                                                               values="score").round(4))
    print(f"\nMeasures, pooled over prompts{' (relative to N|N)' if rel else ''} (mean [95% CI])")
    print(pooled.assign(value=pooled.apply(fmt, axis=1))
          .pivot(index=["family", "contrast"], columns="judge", values="value")
          .reindex(columns=judges).to_string())
    key = stances[(stances["grouping"] == "match") & stances["measure"].isin(
        KEY_MEASURES + ["cand F - M | ref F", "cand F - M | ref M", "N|N - mixed"])]
    print("\nBy stance match (mean [95% CI])")
    print(key.assign(value=key.apply(fmt, axis=1)).pivot(index=["measure", "topic"], columns="judge", values="value")
          .reindex(columns=judges).to_string())
    if stance_tests:
        t = pd.DataFrame(stance_tests)
        print("\nStance heterogeneity, permutation p-value\n",
              t.pivot(index=["grouping", "measure"], columns="judge", values="p_value").reindex(columns=judges)
              .round(4).to_string())
    if group_rows:
        t = pd.DataFrame(group_tests)
        t = t[t["grouping"].isin(["selection", "authority"])]
        print("\nSituation-grouping heterogeneity (selection, authority), permutation p-value\n",
              t.pivot(index=["grouping", "measure"], columns="judge", values="p_value").reindex(columns=judges)
              .round(4).to_string())
    if flip_rows:
        flip_rows = pd.DataFrame(flip_rows)
        flip_rows.to_csv(out_dir / "flips.csv", index=False)
        plot_flips(flip_rows, judges, out_dir, f"Decision flips — {name}")
        show = flip_rows[flip_rows["relation"].isin(["all", f"close (margin <= {CLOSE})", "same over opposite",
                                                     "same over partial"])]
        fmt_rate = lambda r: f"{r['mean']:.1%} [{r['ci_low']:.1%}, {r['ci_high']:.1%}]"  # noqa: E731
        print("\nDecision flips: share of the eligible pairs / rankings that flip (mean [95% CI])")
        print(show.assign(value=show.apply(fmt_rate, axis=1))
              .pivot(index=["unit", "relation", "metric"], columns="judge", values="value")
              .reindex(columns=judges).to_string())
    print("\nPer-situation agreement between judges (Spearman), mean over judge pairs\n",
          agreement.groupby("measure")["spearman"].agg(["mean", "min", "max"]).round(2))
    if len(replication):
        print("\nReplication: both setup vs single-side setup\n",
              replication.assign(both=replication.apply(lambda r: f"{r['both']:+.4f}", axis=1),
                                 single=replication.apply(lambda r: f"{r['single']:+.4f}", axis=1))
              .pivot(index="measure", columns="judge", values=["both", "single"]).to_string())
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
