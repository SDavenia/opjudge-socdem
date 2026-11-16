"""
Score how well each candidate opinion reflects its reference with an LLM judge, on the [0, 1]
alignment scale of opinion_judge_stability_bias, with each of its 5 prompt variations.

A pair is a reference (a human Habermas opinion) and a candidate (a synthetic HBY opinion) of
the same situation: each reference is paired with all the candidates of its situation. The
reference is the first opinion of the prompt and the candidate the second, as the prompts ask
"how well the second opinion aligns with the first".

Setups (--setup), by which side carries the sociodemographic preamble of --variable:
  none       original references x original candidates (baseline)
  reference  hby_reference_{variable} (every level, neutral included) x original candidates
A new setup (e.g. intervened candidates, or both sides) is one more entry of SETUPS: each side
is read and paired the same way, with its level in reference_level / candidate_level
("original" when it is not intervened).

Output, in {output_dir}/{setup}[_{variable}]/:
  {judge}_by_prompt.csv  one row per pair and prompt variation: the judge's answer and its score
                         (no texts: they are in the _mean file and in data/)
  {judge}_mean.csv       one row per pair, with the texts: the mean and std of the score over the
                         prompt variations whose answer has one, and how many those are

Usage:
    python score.py --setup reference --variable gender --judge_model_id gemma3
    python score.py --setup reference --variable gender --judge_model_id gemma3 --num_situations 10
    python score.py --setup none --judge_model_id openai/gpt-5.4-nano
    python score.py --setup reference --variable gender --judge_model_id llama-3.2-1b --backend transformers
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from add_sociodem_preamble import DATA_DIR, SOCIODEMOGRAPHICS, SPLITS
from judges import is_local, run_judge
from prompts import SCORING_PROMPTS

OUTPUT_DIR = Path(__file__).resolve().parent / "scores"

# Setup -> whether each side of the pair carries the preamble.
SETUPS = {
    "none": {"reference": False, "candidate": False},
    "reference": {"reference": True, "candidate": False},
}
ORIGINAL_LEVEL = "original"
SIDE_COLUMNS = ["text_id", "original_text_id", "level", "stance", "stance_group", "text"]
PAIR_KEYS = ["reference_original_text_id", "candidate_original_text_id"]
BY_PROMPT_COLUMNS = (["pair_id", "situation_id"]
                     + [f"{side}_{c}" for side in SPLITS for c in SIDE_COLUMNS if c != "text"]
                     + ["prompt_variation", "answer", "score"])
# "0.75", ".75", "1.0", "0", "1", but not the "1" of "10"
SCORE_RE = re.compile(r"(?<!\d)(?:0?\.\d+|1\.0+|0|1)(?!\d)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup", required=True, choices=list(SETUPS))
    parser.add_argument("--variable", choices=list(SOCIODEMOGRAPHICS),
                        help="Sociodemographic of the preamble, for the setups that intervene on a side.")
    parser.add_argument("--judge_model_id", required=True,
                        help="A local judge of judges.REGISTRY, or an OpenRouter slug (e.g. openai/gpt-5.4-nano).")
    parser.add_argument("--num_situations", type=int, default=None,
                        help="Only the first N situations (in the order of data/hby.csv).")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=1000,
                        help="Save the partial _by_prompt file every N answers (0: only at the end).")
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm", help="Local judges.")
    parser.add_argument("--tensor_parallel_size", type=int, default=None, help="vLLM (default: 1).")
    parser.add_argument("--pipeline_parallel_size", type=int, default=None,
                        help="vLLM (default: the visible GPUs / tensor_parallel_size).")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="vLLM: context length the KV cache is sized for (prompts are under 1k tokens).")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="vLLM.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="transformers: divided by batch_size_divide.")
    parser.add_argument("--max_concurrency", type=int, default=8, help="OpenRouter judges: requests in flight.")
    parser.add_argument("--max_retries", type=int, default=5, help="OpenRouter judges.")
    args = parser.parse_args()
    intervened = any(SETUPS[args.setup].values())
    if intervened and args.variable is None:
        parser.error(f"--setup {args.setup} needs --variable.")
    if not intervened:
        args.variable = None
    if not is_local(args.judge_model_id) and "/" not in args.judge_model_id:
        parser.error(f"--judge_model_id {args.judge_model_id} is neither a local judge of judges.REGISTRY nor an "
                     f"OpenRouter slug (provider/model).")
    if args.num_situations is not None and args.num_situations < 1:
        parser.error("--num_situations must be at least 1.")
    return args


def load_side(side, intervened, variable, data_dir):
    """The opinions of one side, with their columns prefixed by the side."""
    if intervened:
        df = pd.read_csv(data_dir / f"hby_{side}_{variable}.csv").rename(columns={variable: "level"})
    else:
        df = pd.read_csv(data_dir / "hby.csv")
        df = df[df["origin"] == SPLITS[side]].assign(original_text_id=lambda d: d["text_id"], level=ORIGINAL_LEVEL)
    return df[["situation_id", "situation"] + SIDE_COLUMNS].rename(columns={c: f"{side}_{c}" for c in SIDE_COLUMNS})


def build_pairs(args):
    """One row per reference and candidate of the same situation, the versions of each original
    pair (its sociodemographic levels) consecutive."""
    intervened = SETUPS[args.setup]
    references = load_side("reference", intervened["reference"], args.variable, args.data_dir)
    candidates = load_side("candidate", intervened["candidate"], args.variable, args.data_dir).drop(columns="situation")
    if args.num_situations is not None:
        keep = pd.read_csv(args.data_dir / "hby.csv", usecols=["situation_id"])["situation_id"].drop_duplicates()
        keep = keep.head(args.num_situations)
        references = references[references["situation_id"].isin(keep)]
        candidates = candidates[candidates["situation_id"].isin(keep)]
    pairs = references.merge(candidates, on="situation_id")
    pairs.insert(0, "pair_id", pairs["reference_text_id"] + "__" + pairs["candidate_text_id"])
    pairs["_pair"] = pairs.groupby(PAIR_KEYS, sort=False).ngroup()
    return pairs.sort_values("_pair", kind="stable").reset_index(drop=True)


def expand_prompts(pairs):
    """(rows, prompts, group_size): one row and prompt per pair and prompt variation, ordered by
    original pair, then variation, then level, so that the group_size versions of an original
    pair under a variation are consecutive."""
    variations = list(SCORING_PROMPTS)
    rows = pairs.merge(pd.DataFrame({"prompt_variation": variations}), how="cross")
    rows["_variation"] = rows["prompt_variation"].map(variations.index)
    rows = rows.sort_values(["_pair", "_variation"], kind="stable").reset_index(drop=True)
    prompts = [SCORING_PROMPTS[v].format(situation=s, first_opinion=r, second_opinion=c)
               for v, s, r, c in zip(rows["prompt_variation"], rows["situation"], rows["reference_text"],
                                     rows["candidate_text"])]
    sizes = rows.groupby(["_pair", "_variation"]).size().unique()
    return rows, prompts, int(sizes[0]) if len(sizes) == 1 else 1


def parse_score(answer, decimals=2):
    """The [0, 1] score in a judge's answer, or None if there is none or more than one."""
    if not answer:
        return None
    scores = {round(float(m), decimals) for m in SCORE_RE.findall(answer) if 0.0 <= float(m) <= 1.0}
    return scores.pop() if len(scores) == 1 else None


def scored_by_prompt(rows, answers, meta):
    return rows.assign(answer=answers, score=[parse_score(a) for a in answers])[BY_PROMPT_COLUMNS].assign(**meta)


def mean_scores(pairs, by_prompt, meta):
    stats = by_prompt.groupby("pair_id")["score"].agg(score_mean="mean", score_std="std", n_scored_prompts="count")
    return (pairs.drop(columns="_pair").merge(stats.reset_index(), on="pair_id", how="left")
            .assign(n_prompts=len(SCORING_PROMPTS), **meta))


def main():
    args = parse_args()
    pairs = build_pairs(args)
    rows, prompts, group_size = expand_prompts(pairs)
    judge_name = args.judge_model_id.replace("/", "_")
    out_dir = args.output_dir / (args.setup if args.variable is None else f"{args.setup}_{args.variable}")
    out_dir.mkdir(parents=True, exist_ok=True)
    by_prompt_path, mean_path = out_dir / f"{judge_name}_by_prompt.csv", out_dir / f"{judge_name}_mean.csv"
    meta = {"judge_model": args.judge_model_id, "setup": args.setup, "variable": args.variable}
    print(f"Setup {args.setup} | {pairs['situation_id'].nunique()} situations | {len(pairs)} pairs "
          f"({pairs['reference_text_id'].nunique()} references x their candidates) | "
          f"{len(SCORING_PROMPTS)} prompt variations | {len(prompts)} prompts | {group_size} versions per pair")

    last_saved = 0

    def checkpoint(answers, n_done):
        nonlocal last_saved
        if args.save_every > 0 and n_done - last_saved >= args.save_every and n_done < len(answers):
            scored_by_prompt(rows, answers, meta).to_csv(by_prompt_path, index=False)
            last_saved = n_done

    answers = run_judge(args.judge_model_id, prompts, max_new_tokens=args.max_new_tokens, group_size=group_size,
                        backend=args.backend, batch_size=args.batch_size,
                        tensor_parallel_size=args.tensor_parallel_size,
                        pipeline_parallel_size=args.pipeline_parallel_size, max_model_len=args.max_model_len,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        max_concurrency=args.max_concurrency, max_retries=args.max_retries,
                        on_progress=checkpoint)

    by_prompt = scored_by_prompt(rows, answers, meta)
    by_prompt.to_csv(by_prompt_path, index=False)
    mean_scores(pairs, by_prompt, meta).to_csv(mean_path, index=False)
    print(f"Saved {len(by_prompt)} rows to {by_prompt_path} and {len(pairs)} to {mean_path}")
    unscored = by_prompt["score"].isna().groupby(by_prompt["prompt_variation"], sort=False).mean()
    print("Answers without a score, by prompt variation:", unscored.round(3).to_dict())


if __name__ == "__main__":
    main()
