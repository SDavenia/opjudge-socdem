"""
Select the opinions and ValuePrism rows behind the situations of data/selected_situations.csv.

  HB (Habermas Machine): up to --n_opinions (6) real opinions per situation, the references,
     spread as evenly as possible over the stances agree / disagree / neutral (the participant's
     first-stage rating of the question; a stance short of opinions gives its slots to the others),
     and within a stance over the fine-grained ratings (AGREE vs SOMEWHAT_AGREE, ...). The
     situation is question.text (the question), not question.affirming_statement.
  VP (ValuePrism): up to --n_opinions (6) rows (vrd, value, explanation, valence) per situation,
     spread as evenly as possible over the valences Supports / Opposes / Either, and within a
     valence as different from each other as possible (token Jaccard distance of the
     explanations). generate_opinions.py writes one opinion from each.

As in opinion_judge_stability_bias (prepare_data.py), and with the same opinion filters: Habermas
opinions written by participants (not BOT_CITIZEN), completed, one per participant and question,
with the first-stage position rating.

Output (situation columns of selected_situations.csv carried as selection_topic,
expected_authority, gender_role_reference, heteronormative):
  data/hb_human.csv     one row per selected Habermas opinion (origin == human), with the
                        Habermas topic columns
  data/vp_selected.csv  one row per selected ValuePrism row (no opinion text yet)

Usage:
    python prepare_data.py                                   # downloads the Habermas parquets (~500 MB)
    python prepare_data.py --habermas_dir path/to/parquets   # local copies
"""
import argparse
import logging
import re
import urllib.request
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from topics import TOPIC_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
log = logging.getLogger("prepare_data")

DATA_DIR = Path(__file__).resolve().parent / "data"
HABERMAS_URL = "https://storage.googleapis.com/habermas_machine/datasets/{}"
HABERMAS_RATINGS_FILE = "hm_all_position_statement_ratings.parquet"
HABERMAS_CANDIDATES_FILE = "hm_all_candidate_comparisons.parquet"
VALUEPRISM_URL = "hf://datasets/allenai/ValuePrism/full/full.csv"

HB_STANCES = ["agree", "disagree", "neutral"]
HB_STANCE_GROUPS = {"STRONGLY_AGREE": "agree", "AGREE": "agree", "SOMEWHAT_AGREE": "agree", "NEUTRAL": "neutral",
                    "SOMEWHAT_DISAGREE": "disagree", "DISAGREE": "disagree", "STRONGLY_DISAGREE": "disagree"}
VP_STANCES = ["Supports", "Opposes", "Either"]
VP_STANCE_GROUPS = {"Supports": "agree", "Opposes": "disagree", "Either": "neutral"}

# selected_situations.csv column -> output column.
SELECTION_COLUMNS = {"topic": "selection_topic", "Expected more authority": "expected_authority",
                     "Contains gender role reference": "gender_role_reference", "eteronormativo": "heteronormative"}
# question.id of selected_situations.csv that is not in the Habermas data -> the Habermas id of the
# same question.text ("Should all employers be required to offer flexible working arrangements to
# employees with childcare responsibilities?").
HB_ID_FIXES = {"S173961813": "S173965963"}
# Situations whose human stance labels do not refer to question.text: the Habermas data pairs
# "Is Manchester United a better football team than Liverpool?" with the affirming statement
# "Football is a more interesting sport than rugby.", which the participants rated.
HB_STANCE_LABEL_MISMATCH = {"HB_173962115"}


# --- Selection helpers (from opinion_judge_stability_bias/prepare_data.py) ---------------------

def balanced_allocation(capacities, target_total):
    """Split target_total slots over the groups as evenly as possible, a group that cannot fill
    its share giving the shortfall to the others."""
    allocation = {g: 0 for g in capacities}
    remaining = min(target_total, sum(capacities.values()))
    active = [g for g in capacities if capacities[g] > 0]
    while remaining > 0 and active:
        share, extra = divmod(remaining, len(active))
        if share == 0:
            for g in sorted(active, key=lambda g: -(capacities[g] - allocation[g]))[:extra]:
                allocation[g] += 1
                remaining -= 1
            break
        for g in active:
            take = min(share, capacities[g] - allocation[g])
            allocation[g] += take
            remaining -= take
        active = [g for g in active if capacities[g] - allocation[g] > 0]
    return allocation


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _jaccard_distance(a, b):
    return 1 - len(a & b) / len(a | b) if a | b else 0.0


def pick_diverse(texts, k):
    """Indices of k texts as different from each other as possible (greedy farthest point)."""
    n = len(texts)
    if n <= k:
        return list(range(n))
    if k <= 0:
        return []
    if k == 1:
        return [0]
    toks = [_tokens(t) for t in texts]
    dist = [[_jaccard_distance(toks[i], toks[j]) for j in range(n)] for i in range(n)]
    selected = list(max(combinations(range(n), 2), key=lambda p: dist[p[0]][p[1]]))
    while len(selected) < k:
        selected.append(max((i for i in range(n) if i not in selected),
                            key=lambda i: min(dist[i][s] for s in selected)))
    return selected


def pick_diverse_sublabels(group, label_col, k, random_state=0):
    """k rows of group, one per distinct label before any label repeats (shuffled order)."""
    if len(group) <= k:
        return group
    shuffled = group.sample(frac=1, random_state=random_state)
    first = shuffled.drop_duplicates(label_col)
    chosen = list(first.index[:k])
    chosen += [i for i in shuffled.index if i not in chosen][:k - len(chosen)]
    return group.loc[chosen]


# --- Selected situations -----------------------------------------------------------------------

def load_selected(path):
    sel = pd.read_csv(path, dtype=str)
    sel.columns = sel.columns.str.strip()
    sel["question.id"] = sel["question.id"].str.strip().replace(HB_ID_FIXES)
    sel["question.text"] = sel["question.text"].str.strip()
    n = len(sel)
    sel = sel.drop_duplicates("question.id")
    if len(sel) < n:
        log.info("Dropped %d duplicated situations of %s", n - len(sel), path)
    sel = sel.rename(columns=SELECTION_COLUMNS)
    for col in ["gender_role_reference"]:
        sel[col] = sel[col].fillna("").str.strip().str.lower().eq("x")
    sel["heteronormative"] = sel["heteronormative"].notna()
    return sel


# --- Habermas ----------------------------------------------------------------------------------

def load_habermas(habermas_dir):
    def read(name, tmp):
        path = (habermas_dir or tmp) / name
        if not path.exists():
            log.info("Downloading %s", name)
            urllib.request.urlretrieve(HABERMAS_URL.format(name), path)
        return pd.read_parquet(path)

    with TemporaryDirectory() as tmp:
        ratings, candidates = read(HABERMAS_RATINGS_FILE, Path(tmp)), read(HABERMAS_CANDIDATES_FILE, Path(tmp))

    keys = ["metadata.participant_id", "question.id", "question.text", "question.affirming_statement",
            "question.negating_statement"]
    ratings = ratings[ratings["rating_index"] == 0][keys + ["ratings.agreement"]].copy()
    ratings["ratings.agreement"] = ratings["ratings.agreement"].apply(lambda x: x[0])
    ratings = ratings[ratings["ratings.agreement"] != "[MOCK]"]

    candidates = candidates[(candidates["own_opinion.metadata.provenance"] != "BOT_CITIZEN")
                            & (candidates["own_opinion.metadata.status"] == "COMPLETED")]
    candidates = candidates.drop_duplicates(subset=["metadata.participant_id", "question.id"])
    candidates = candidates[keys + ["own_opinion.text"]]
    return ratings.merge(candidates, on=keys)


def prepare_hb(selected, habermas_dir, n_opinions, topics_path):
    sel = selected[selected["source"] == "HB"]
    hm = load_habermas(habermas_dir)
    hm = hm[hm["question.id"].isin(sel["question.id"])].copy()
    hm["own_opinion.text"] = hm["own_opinion.text"].str.strip()
    hm = hm[hm["own_opinion.text"].fillna("") != ""]
    hm["stance_group"] = hm["ratings.agreement"].map(HB_STANCE_GROUPS)
    hm = hm.dropna(subset=["stance_group"])

    parts = []
    for qid, g in hm.groupby("question.id", sort=False):
        capacities = {s: int((g["stance_group"] == s).sum()) for s in HB_STANCES}
        for stance, k in balanced_allocation(capacities, n_opinions).items():
            if k:
                parts.append(pick_diverse_sublabels(g[g["stance_group"] == stance], "ratings.agreement", k))
    hb = pd.concat(parts)

    hb["situation_id"] = hb["question.id"].str.replace("S", "HB_", regex=False)
    hb["text_id"] = "HBH_" + hb["situation_id"].str.removeprefix("HB_") + "_" + hb["metadata.participant_id"]
    hb = hb.merge(sel, on="question.id", how="left", suffixes=("", "_selected"))
    hb = hb.rename(columns={"question.text": "situation", "question.affirming_statement": "hb_affirming_statement",
                            "ratings.agreement": "stance", "metadata.participant_id": "hb_participant_id",
                            "own_opinion.text": "text"})
    topics = pd.read_csv(topics_path)[["situation_id"] + TOPIC_COLUMNS]
    hb = hb.merge(topics, on="situation_id", how="left")
    hb["stance_label_mismatch"] = hb["situation_id"].isin(HB_STANCE_LABEL_MISMATCH)
    hb = hb.assign(dataset="hb", origin="human", generation_model="human", generation_prompt_version=None)
    # selected_situations order, then stance
    order = {q: i for i, q in enumerate(sel["question.id"])}
    hb = hb.assign(_o=hb["question.id"].map(order), _s=hb["stance_group"].map(HB_STANCES.index))
    hb = hb.sort_values(["_o", "_s"], kind="stable")
    cols = (["situation_id", "text_id", "situation"] + TOPIC_COLUMNS + list(SELECTION_COLUMNS.values())
            + ["hb_affirming_statement", "stance_label_mismatch", "stance", "stance_group", "hb_participant_id", "text",
               "dataset", "origin", "generation_model", "generation_prompt_version"])

    missing = set(sel["question.id"]) - set(hb["question.id"])
    if missing:
        log.warning("[hb] %d selected situations have no opinion: %s", len(missing), sorted(missing))
    return hb[cols]


# --- ValuePrism --------------------------------------------------------------------------------

def prepare_vp(selected, valueprism_path, vp_situations_path, n_opinions):
    sel = selected[selected["source"] == "VP"]
    ids = pd.read_csv(vp_situations_path).rename(columns={"id_situation": "situation_id"})
    vp = pd.read_csv(valueprism_path, dtype={"vrd": str, "text": str})
    # The VPS_ ids number the situations in their order of appearance in ValuePrism.
    assert vp["situation"].drop_duplicates().reset_index(drop=True).equals(ids["situation"]), \
        f"{vp_situations_path} does not number the situations of {valueprism_path}"
    vp = vp.merge(ids, on="situation")
    vp = vp[vp["situation_id"].isin(sel["question.id"])]

    parts = []
    for sid, g in vp.groupby("situation_id", sort=False):
        capacities = {s: int((g["valence"] == s).sum()) for s in VP_STANCES}
        for stance, k in balanced_allocation(capacities, n_opinions).items():
            if k:
                group = g[g["valence"] == stance]
                parts.append(group.iloc[pick_diverse(group["explanation"].fillna("").tolist(), k)])
    out = pd.concat(parts)
    out["text_id"] = [f"VP_{i}" for i in out.index]  # row of ValuePrism full.csv
    out = out.rename(columns={"valence": "stance", "vrd": "vp_vrd", "text": "vp_vrd_value",
                              "explanation": "vp_explanation"})
    out["stance_group"] = out["stance"].map(VP_STANCE_GROUPS)
    out = out.merge(sel.drop(columns=["question.text", "source"]), left_on="situation_id", right_on="question.id")
    out = out.assign(dataset="vp", origin="synthetic")
    order = {q: i for i, q in enumerate(sel["question.id"])}
    out = out.assign(_o=out["situation_id"].map(order), _s=out["stance"].map(VP_STANCES.index))
    out = out.sort_values(["_o", "_s"], kind="stable")
    cols = (["situation_id", "text_id", "situation"] + list(SELECTION_COLUMNS.values())
            + ["stance", "stance_group", "vp_vrd", "vp_vrd_value", "vp_explanation", "dataset", "origin"])

    missing = set(sel["question.id"]) - set(out["situation_id"])
    if missing:
        log.warning("[vp] %d selected situations are not in ValuePrism: %s", len(missing), sorted(missing))
    return out[cols]


def log_counts(df, name, n_opinions):
    counts = df.groupby(["situation_id", "stance_group"]).size().unstack(fill_value=0).reindex(
        columns=HB_STANCES, fill_value=0)
    total = counts.sum(axis=1)
    log.info("[%s] %d rows over %d situations; %d with fewer than %d; per stance group: %s", name, len(df),
             len(counts), (total < n_opinions).sum(), n_opinions, counts.sum().to_dict())
    short = counts[total < n_opinions]
    if len(short):
        log.info("[%s] situations with fewer than %d:\n%s", name, n_opinions, short.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selected", type=Path, default=DATA_DIR / "selected_situations.csv")
    parser.add_argument("--habermas_dir", type=Path, default=None,
                        help="Directory with the two Habermas parquets (default: download them).")
    parser.add_argument("--valueprism", default=VALUEPRISM_URL, help="ValuePrism full.csv (path or hf:// URL).")
    parser.add_argument("--vp_situations", type=Path, default=DATA_DIR / "vp_situations.csv")
    parser.add_argument("--topics", type=Path, default=DATA_DIR / "hb_situation_topics.csv")
    parser.add_argument("--n_opinions", type=int, default=6, help="Opinions (HB) / rows (VP) per situation.")
    parser.add_argument("--output_dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    selected = load_selected(args.selected)
    log.info("Selected situations: %s", selected["source"].value_counts().to_dict())
    hb = prepare_hb(selected, args.habermas_dir, args.n_opinions, args.topics)
    vp = prepare_vp(selected, args.valueprism, args.vp_situations, args.n_opinions)
    for df, name in [(hb, "hb_human"), (vp, "vp_selected")]:
        log_counts(df, name, args.n_opinions)
        path = args.output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        log.info("Wrote %s", path)


if __name__ == "__main__":
    main()
