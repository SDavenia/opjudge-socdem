"""
Generate the synthetic opinions of the selected situations (prepare_data.py), as in
opinion_judge_stability_bias (prepare_habermas_hybrid.py, prepare_valueprism_generations.py):

  HB  one opinion per situation and stance (agree / disagree / neutral), the candidates, from the
      question (question.text) and a stance instruction (prompts.HB_STANCE_INSTRUCTIONS), with the
      prompt hb_reflective_person. Written with the human opinions (the references) to data/hb.csv.
  VP  one opinion per selected ValuePrism row, from the situation, its value, valence and
      explanation (prompts.VP_STANCE_INSTRUCTIONS), with the prompt reflective_person. Written to
      data/vp.csv; there are no references, so score.py pairs these opinions with each other.

Sampled (temperature 0.8, top_p 0.9, as the original scripts) with vLLM, the model split over
the visible GPUs by pipeline parallelism (see judges.run_vllm), with a fixed seed.

Usage:
    python generate_opinions.py --generation_model_id llama-3.3-70b
    python generate_opinions.py --generation_model_id llama-3.2-1b --datasets hb --num_situations 2   # smoke test
"""
import argparse
import os
from pathlib import Path

import pandas as pd

from judges import REGISTRY
from prepare_data import DATA_DIR, HB_STANCES
from prompts import (GENERATION_PROMPTS, HB_STANCE_INSTRUCTIONS, HB_STANCE_INSTRUCTIONS_BY_SITUATION,
                     VP_STANCE_INSTRUCTIONS)

HB_PROMPT, VP_PROMPT = "hb_reflective_person", "reflective_person"
# HB synthetic stance_group -> Likert stance label (a valid human-side label too).
HB_SYNTHETIC_STANCES = {"agree": "AGREE", "disagree": "DISAGREE", "neutral": "NEUTRAL"}


def hb_prompts(human):
    """One row per situation and stance, with its prompt."""
    situations = human.drop_duplicates("situation_id")
    situation_cols = [c for c in situations.columns
                      if c not in {"text_id", "stance", "stance_group", "hb_participant_id", "text", "origin",
                                   "generation_model", "generation_prompt_version"}]
    rows = []
    for sit in situations[situation_cols].to_dict("records"):
        instructions = HB_STANCE_INSTRUCTIONS_BY_SITUATION.get(sit["situation_id"], HB_STANCE_INSTRUCTIONS)
        for group in HB_STANCES:
            rows.append({**sit, "text_id": f"HBS_{sit['situation_id'].removeprefix('HB_')}_{group}",
                         "stance": HB_SYNTHETIC_STANCES[group], "stance_group": group,
                         "stance_instruction": instructions[group],
                         "prompt": GENERATION_PROMPTS[HB_PROMPT].format(situation=sit["situation"],
                                                                        stance_instruction=instructions[group])})
    return pd.DataFrame(rows)


def vp_prompts(selected):
    df = selected.copy()
    df["stance_instruction"] = [VP_STANCE_INSTRUCTIONS[s].format(value=v)
                                for s, v in zip(df["stance"], df["vp_vrd_value"])]
    df["prompt"] = [GENERATION_PROMPTS[VP_PROMPT].format(situation=s, value=v, valence=st, explanation=e,
                                                         stance_instruction=i)
                    for s, v, st, e, i in zip(df["situation"], df["vp_vrd_value"], df["stance"],
                                              df["vp_explanation"], df["stance_instruction"])]
    return df


def generate(spec, prompts, *, max_new_tokens, temperature, top_p, seed, tensor_parallel_size, pipeline_parallel_size,
             max_model_len, gpu_memory_utilization):
    """(texts, finish_reasons), one per prompt, in order: the model's chat template, sampled."""
    import torch

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # see judges.run_vllm
    from vllm import LLM, SamplingParams

    tp = tensor_parallel_size or 1
    pp = pipeline_parallel_size or max(1, torch.cuda.device_count() // tp)
    kwargs = {}
    if spec.slow_tokenizer:
        kwargs["tokenizer_mode"] = "mistral"
    if spec.is_vlm:
        kwargs["limit_mm_per_prompt"] = {"image": 0}
    llm = LLM(model=spec.name, dtype="bfloat16", tensor_parallel_size=tp, pipeline_parallel_size=pp,
              max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization, seed=seed, **kwargs)
    params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, seed=seed)
    print(f"Generator {spec.name} (vLLM, tensor parallel {tp}, pipeline parallel {pp}) | {len(prompts)} prompts")
    outputs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params)
    return [o.outputs[0].text.strip() for o in outputs], [o.outputs[0].finish_reason for o in outputs]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generation_model_id", required=True, choices=list(REGISTRY))
    parser.add_argument("--datasets", nargs="+", choices=["hb", "vp"], default=["hb", "vp"])
    parser.add_argument("--num_situations", type=int, default=None, help="Only the first N situations of each dataset.")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--pipeline_parallel_size", type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    def head(df):
        if args.num_situations is None:
            return df
        return df[df["situation_id"].isin(df["situation_id"].drop_duplicates().head(args.num_situations))]

    human = head(pd.read_csv(args.data_dir / "hb_human.csv"))
    todo = {}
    if "hb" in args.datasets:
        todo["hb"] = hb_prompts(human)
    if "vp" in args.datasets:
        todo["vp"] = vp_prompts(head(pd.read_csv(args.data_dir / "vp_selected.csv")))

    prompts = [p for df in todo.values() for p in df["prompt"]]
    texts, finish = generate(REGISTRY[args.generation_model_id], prompts, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_p=args.top_p, seed=args.seed,
                             tensor_parallel_size=args.tensor_parallel_size,
                             pipeline_parallel_size=args.pipeline_parallel_size, max_model_len=args.max_model_len,
                             gpu_memory_utilization=args.gpu_memory_utilization)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = 0
    for name, df in todo.items():
        n = len(df)
        df = df.drop(columns="prompt").assign(
            text=texts[start:start + n], finish_reason=finish[start:start + n], origin="synthetic",
            generation_model=args.generation_model_id, generation_prompt_version=HB_PROMPT if name == "hb" else VP_PROMPT,
            generation_temperature=args.temperature, generation_top_p=args.top_p, generation_seed=args.seed)
        start += n
        if name == "hb":  # references first, then candidates, as in hby.csv
            df = pd.concat([human, df], ignore_index=True)
        path = args.output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        synth = df[df["origin"] == "synthetic"]
        print(f"[{name}] wrote {len(df)} rows ({len(synth)} generated, {(synth['finish_reason'] == 'length').sum()} "
              f"cut at --max_new_tokens) to {path}")


if __name__ == "__main__":
    main()
