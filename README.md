# Sociodemographic bias in opinion judges

Does an LLM-as-a-judge change the alignment score between a reference and a candidate opinion
when the only difference is a sociodemographic cue about the speaker? If it does, a judge used
as a reward for opinion alignment would push the aligned model toward the groups it favours.

## Design

Each item is a situation `s`, a reference opinion `r` and a candidate opinion `c`. We intervene
on `r`, on `c`, or on both (2x2):

| | candidate: original | candidate: intervened |
|---|---|---|
| **reference: original** | baseline | candidate intervention: is one group given more credibility? |
| **reference: intervened** | reference intervention: is the reference read through a stereotype? | double intervention: same vs. different group |

Each variable is tested on its own, with an explicit preamble ("As a woman, ..."). Every
intervened opinion also has a **neutral** level ("As a person, ...") that adds the same preamble
with no sociodemographic. Use it, not the original opinion, as the baseline for the levels: the
length and framing of the preamble are then the same.

Pilot: **gender** (female, male), explicit cue, pattern-based preamble.

## Data

`data/hby.csv` is Habermas-hybrid (HBY), copied unchanged from
`opinion_judge_stability_bias/artifacts/generations/habermas_hybrid/llama-3.3-70b_hb_reflective_person.csv`:
500 Habermas Machine situations, 2686 human opinions (`origin == human`) and one synthetic
opinion per situation and stance by Llama-3.3-70B (`origin == synthetic`, 1500).

| File | Content |
|---|---|
| `data/hby.csv` | original HBY |
| `data/hby_reference_{variable}.csv` | the human opinions (references), one row per opinion and level |
| `data/hby_candidate_{variable}.csv` | the synthetic opinions (candidates), one row per opinion and level |

The intervened files keep all the HBY columns, with `text` holding the intervened opinion and
`text_id` made unique as `{original_text_id}_{variable}_{level}`, so they can be paired by
`situation_id` like HBY. Added columns:

| Column | |
|---|---|
| `{variable}` | level: `female`, `male` or `neutral` |
| `identity_phrase` | what fills the preamble: `a woman`, `a man`, `a person` |
| `original_text_id`, `original_text` | the HBY opinion |
| `preamble_frame` | the opinion with an `{identity}` slot, shared by all the levels |
| `preamble_method` | `pattern` (an LLM-based variant would write `llm`) |
| `preamble_rule` | which rule placed the slot (see below) |
| `{variable}_cue_in_original` | a cue of the variable already in the opinion, which the preamble may contradict (2 references for gender) |

## Adding the preamble

```bash
python add_sociodem_preamble.py --variable gender
```

Each opinion is first turned into a frame with an `{identity}` slot, without looking at the
level, and the frame is then filled with each identity phrase. The versions of an opinion thus
differ only in the identity phrase, and the opinion text is kept verbatim (only the case of its
first letter may change). Rules (`preamble_rule`, `marker+` when after a discourse marker):

| Rule | Original | Intervened |
|---|---|---|
| `marker+…` | Honestly, I think ... | Honestly, as a woman, I think ... |
| `first_person` | I don't think ... | As a woman, I don't think ... |
| `subordinate` | While I agree, ... | As a woman, while I agree, ... |
| `first_person_near` | It is my opinion that ... | As a woman, it is my opinion that ... |
| `bridge` | The government should ... | As a woman, I think the government should ... |

`bridge` adds "I think" so that the preamble does not dangle on a non-first-person subject; it
is used when there is no first person in the first 8 words of the first sentence. All 1500
candidates start with "Honestly,", so they all get "Honestly, as a woman, ...".

New variables go in `SOCIODEMOGRAPHICS` in `add_sociodem_preamble.py`: their levels, as phrases
that fit "As ___," (e.g. "someone aged 18-24"), and a pattern of the cues already in the text.

## Scoring

```bash
pip install -r requirements.txt
cp .env.example .env        # OpenRouter key, only for OpenRouter judges

python score.py --setup reference --variable gender --judge_model_id gemma3
python score.py --setup reference --variable gender --judge_model_id gemma3 --num_situations 10   # pilot
python score.py --setup none --judge_model_id openai/gpt-5.4-nano                                  # baseline
```

A pair is a reference and a candidate of the same situation: each reference is paired with all
the candidates of its situation (the 3 synthetic ones, one per stance). The judge scores how well
the candidate reflects the reference on the [0, 1] scale of opinion_judge_stability_bias, with its
5 prompt variations (`prompts.py`, copied verbatim): `base`, `lex_var`, `reverse_order`,
`lex_var_reverse_order`, `criteria_after`. The reference is the first opinion of the prompt,
the candidate the second.

| Setup | References | Candidates | Pairs | Prompts |
|---|---|---|---|---|
| `none` | original | original | 8,058 | 40,290 |
| `reference` | `hby_reference_{variable}` (all levels) | original | 24,174 | 120,870 |

Other setups (intervened candidates, both sides) are one more entry of `SETUPS` in `score.py`.
Judges: the local models of `judges.REGISTRY` (transformers, greedy), or any OpenRouter slug.
With local judges, the versions of a pair that differ only in the sociodemographic level go in
the same batch, so their scores cannot differ because of the batching.

Output in `scores/{setup}[_{variable}]/`:

| File | Rows | Columns |
|---|---|---|
| `{judge}_by_prompt.csv` | pair x prompt variation | ids, levels and stances of both sides, `prompt_variation`, `answer`, `score` (empty if the answer has no single score in [0, 1]) |
| `{judge}_mean.csv` | pair | the same pair columns plus the situation and both texts, `score_mean`, `score_std`, `n_scored_prompts` (of `n_prompts`) |

`reference_level` / `candidate_level` is the sociodemographic level of each side (`original` if
not intervened); `*_original_text_id` links a version back to the HBY opinion.
