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

## Data: selected situations (HB + VP)

The study now uses the situations of `data/selected_situations.csv`: 39 Habermas Machine (HB)
questions and 43 ValuePrism (VP) situations, with their selection columns (`selection_topic`,
`expected_authority`, `gender_role_reference`, `heteronormative`). The HB situation is the
question (`question.text`, "Should we ban ...?"), both in the generation and the scoring
prompts, not `question.affirming_statement` (kept as `hb_affirming_statement` for reference).

```bash
python prepare_data.py --valueprism path/to/ValuePrism/full.csv      # selection (CPU)
python generate_opinions.py --generation_model_id llama-3.3-70b      # generation (GPU, vLLM)
python add_sociodem_preamble.py --variable gender --dataset hb
python add_sociodem_preamble.py --variable gender --dataset vp
python score.py --dataset hb --setup reference --variable gender --judge_model_id gemma3
python score.py --dataset vp --setup reference --variable gender --judge_model_id gemma3
```

| Dataset | References | Candidates | Pairs |
|---|---|---|---|
| `hb` | up to 6 human Habermas opinions per question, as balanced as possible over agree / disagree / neutral (a stance short of opinions gives its slots to the others; within a stance, different fine-grained ratings first) | 3 synthetic opinions per question, one per stance (`hb_reflective_person`, adapted to questions: answers YES / NO / stays neutral) | reference x candidate of the same question |
| `vp` | 6 synthetic opinions per situation, one per ValuePrism row (vrd, value, explanation), the rows as balanced as possible over Supports / Opposes / Either and, within a valence, as different as possible (`reflective_person`) | the same opinions | every ordered pair of two different opinions of a situation |

Generation as in opinion_judge_stability_bias (`prepare_habermas_hybrid.py`,
`prepare_valueprism_generations.py`): Llama-3.3-70B, temperature 0.8, top_p 0.9, seed 0, up to
256 new tokens (`finish_reason` says if one was cut). Files: `data/hb_human.csv` and
`data/vp_selected.csv` (the selection), `data/hb.csv` and `data/vp.csv` (with the opinions,
`origin` human / synthetic), `data/{hb,vp}_{reference,candidate}_{variable}.csv` (with the preamble;
for vp the two hold the same opinions).

Notes on the selection:
- `S173961813` of the selection is not in the Habermas data; the same question text is
  `S173965963`, used instead. `S173964089` is listed twice (once with a leading space).
- 19 HB questions have fewer than 6 human opinions (down to 1, `HB_173961812`), and 4 VP
  situations fewer than 6 ValuePrism rows (down to 3, `VPS_79`).
- `HB_173966814` ("... an increase or decrease in the number of road accidents?") has no yes/no
  answer: its candidates are told INCREASE (agree, as its affirming statement) / DECREASE / neutral
  (`prompts.HB_STANCE_INSTRUCTIONS_BY_SITUATION`).
- `HB_173962115` ("Is Manchester United a better football team than Liverpool?") has the affirming
  statement "Football is a more interesting sport than rugby." in the Habermas data, which is what
  its participants rated: its human stance labels do not refer to the question
  (`stance_label_mismatch`).

## Data: pilot (HBY)

`data/hby.csv` is Habermas-hybrid (HBY), copied (plus the topic columns, see below) from
`opinion_judge_stability_bias/artifacts/generations/habermas_hybrid/llama-3.3-70b_hb_reflective_person.csv`:
500 Habermas Machine situations, 2686 human opinions (`origin == human`) and one synthetic
opinion per situation and stance by Llama-3.3-70B (`origin == synthetic`, 1500).

| File | Content |
|---|---|
| `data/hby.csv` | original HBY |
| `data/hby_reference_{variable}.csv` | the human opinions (references), one row per opinion and level |
| `data/hby_candidate_{variable}.csv` | the synthetic opinions (candidates), one row per opinion and level |
| `data/hb_topic_taxonomy.json` | the topic taxonomy: `topics` (id -> label, focus, subdomain, domain) and `domains` (domain -> subdomains -> topic ids) |
| `data/hb_situation_topics.csv` | all 2280 Habermas situations with their topic columns |

### Topics

Every situation has a Habermas Machine topic (`question.topic`, an integer 0-104), grouped into
35 subdomains and 10 domains. `topics.py` (the taxonomy is `TAXONOMY` there) writes the two
`hb_*` files above and adds these columns, right after `situation`, to `data/hby*.csv` and to
every CSV in `scores/`; `score.py` carries them into new score files:

| Column | |
|---|---|
| `topic` | Habermas topic id (0-104) |
| `topic_label` | e.g. `Transport Policy & Urban Mobility` |
| `topic_subdomain` | e.g. `Transportation and Infrastructure` |
| `topic_domain` | e.g. `Environment, Energy and Infrastructure` |

```bash
python topics.py                     # downloads the Habermas candidate comparisons (~400 MB)
python topics.py --parquet path/to/hm_all_candidate_comparisons.parquet
```

It is idempotent: rerun it after adding data or score files written before the topic columns.

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
pip install -r requirements.txt && pip uninstall -y torchcodec   # see requirements.txt
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
| `candidate` | original | `hby_candidate_{variable}` (all levels) | 24,174 | 120,870 |
| `both` | `hby_reference_{variable}` (all levels) | `hby_candidate_{variable}` (all levels), every reference x candidate level | 72,522 | 362,610 |

With `--dataset hb` / `vp` the same setups give 609 / 1,236 pairs (`none`), 1,827 / 3,708
(`reference`, `candidate`) and 5,481 / 11,124 (`both`). `commands/score_{hb,vp,hby}.sh` run the
gender setups for the four judges (logs in `out/`).

A new setup is one more entry of `SETUPS` in `score.py`.
Judges: the local models of `judges.REGISTRY`, or any OpenRouter slug. Local judges run with
vLLM by default (`--backend vllm`), split over all the visible GPUs by pipeline parallelism (the
GPUs have no P2P link, so the all-reduces of tensor parallelism are slow: Gemma 3 27B is ~8x
faster with `--pipeline_parallel_size 4` than with `--tensor_parallel_size 4`). Both backends
decode greedily, with the `repetition_penalty` of the model's `generation_config.json` (only the
Qwen2.5 models have one), which transformers' `generate` applies even without sampling.

vLLM batches continuously, so the versions of a pair are sent together but their scores can
still differ by the numerical noise of the batch they end up in (about 3% of the answers change
between two parallel layouts). `--backend transformers` puts them in the same padded batch, so
that batching cannot change them, at ~7x the run time.

Output in `scores/{setup}[_{variable}]/`:

| File | Rows | Columns |
|---|---|---|
| `{judge}_by_prompt.csv` | pair x prompt variation | ids, levels and stances of both sides, `prompt_variation`, `answer`, `score` (empty if the answer has no single score in [0, 1]) |
| `{judge}_mean.csv` | pair | the same pair columns plus the situation and both texts, `score_mean`, `score_std`, `n_scored_prompts` (of `n_prompts`) |

`reference_level` / `candidate_level` is the sociodemographic level of each side (`original` if
not intervened); `*_original_text_id` links a version back to the HBY opinion.

## Analysis

```bash
python analyze.py scores/reference_gender      # -> analysis/hb/reference_gender/
```

For every judge of the directory, how much the score moves when only the prompt variation
changes (level fixed) vs. when only the level changes (prompt fixed): mean absolute difference
over the pairs of conditions and share of pairs whose scores differ, for the prompt, the levels
(female vs male) and the cue (each level vs neutral); and the signed gaps (female - neutral,
male - neutral, female - male), pooled and by prompt variation. 95% CIs by bootstrap over
situations. Writes `variation.csv`, `signed_gaps.csv`, `mean_scores.csv` and a plot of each.

By situation (`situations.csv`): every gap and unsigned measure averaged over the situation's
pairs and prompts, with a standard error clustered by reference opinion. `top_situations.csv` /
`.png` rank the situations by the female - male gap averaged over the judges (`--top` of them
plotted), and `situation_agreement.csv` is the Spearman correlation of those gaps between judges.

By topic (needs the topic columns, see Topics): `topics.csv` has every signed gap and the levels/cue
mean absolute differences by topic domain (10) and subdomain (34), with 95% CIs by bootstrap over
the topic's situations; `topics_{domain,subdomain}.png` plot the signed gaps of the groups with at
least `--min_situations` situations (default 5). The topics (103 labels, median 4 situations) are
too small for this. `topic_heterogeneity.csv` tests whether the topics differ more than random
groups of situations of the same sizes: eta2 (the between-topic share of the variance of the
per-situation values), its expected value with no topic effect, (G - 1) / (N - 1), and a
permutation p-value (`--n_perm`, default 2000). There are 30 such tests per run (3 judges x 2
groupings x 5 measures), so correct for multiple comparisons before reading single p-values.

Groupings (`topics.csv`, `topic_heterogeneity.csv`, `topics_{grouping}.png`): every one of
`selection_topic` (`selection`), `expected_authority` (`authority`), `topic_domain`, `topic_subdomain`
that the scores have with at least 2 values.

Stereotype congruence (`congruent - incongruent` in `signed_gaps*.csv`, gender only): the score of the
level the stereotype expects in the situation (`expected_authority`: M -> male, W -> female) minus the
other one, averaged with the M and W situations weighted equally (bootstrap stratified by
`expected_authority`). A gap shared by all situations (e.g. female > male everywhere) then cancels:
it equals ((female - male | W) - (female - male | M)) / 2. Positive = the judge favours the
stereotype-expected gender. Not in the topic tables (within a group it is just +-(female - male)).

```bash
python analyze.py scores/hb_reference_gender --out_dir analysis/hb/reference_gender
python analyze.py scores/hb_reference_gender --out_dir analysis/hb/reference_gender --relative
python analyze.py scores/hb_candidate_gender --out_dir analysis/hb/candidate_gender
python analyze.py scores/hb_candidate_gender --out_dir analysis/hb/candidate_gender --relative
```

Double intervention (`both`): `analyze_both.py` analyses the 3 x 3 cells of reference x candidate level
(baseline N|N = "As a person" on both sides): main effects, interactions, gender matching net of the
lexical overlap of identical phrases (`match excess` = (same - cross) - (N|N - mixed)), F|F vs M|M,
the candidate female - male gap by reference level, stereotype congruence, the same stance and topic
breakdowns, decision flips (does a cue make a lower-ranked candidate overtake?) and the replication of
the single-side effects. See its docstring.

```bash
python analyze_both.py scores/hb_both_gender --out_dir analysis/hb/both_gender
python analyze_both.py scores/hb_both_gender --out_dir analysis/hb/both_gender --relative
```

VP and joint (HB + VP pooled) analyses: the same scripts. Several score directories are concatenated (the
situation ids differ between datasets). The topic groupings that only one dataset has (`topic_domain`,
`topic_subdomain`: topics.py was run on HB only) are dropped, and a `dataset` grouping (hb vs vp) is added.
The lmm scripts take comma-separated directories.

Outputs are laid out as `analysis/<dataset>/<setup>/`, with `<dataset>` = `hb`, `vp` or `joint`
(the default `--out_dir` follows it: `scores/vp_reference_gender` -> `analysis/vp/reference_gender`,
`scores/hb_* scores/vp_*` -> `analysis/joint/*`). `run_analysis.sh` runs everything (single-side,
both, absolute and relative, and the candidate LMM in `candidate_gender/lmm/`) for the given datasets:

```bash
./run_analysis.sh               # hb vp joint
./run_analysis.sh vp            # analysis/vp/{reference,candidate,both}_gender
```

Write-ups of the results: `docs/reference_gender.md`, `docs/candidate_gender.md`, `docs/both_gender.md`.

By stance (`stances*.csv`, `stances_{reference,pair,match}*.png`): every signed gap (with the congruence
contrast) and unsigned mad by the reference's stance group (agree / neutral / disagree), by the pair
reference -> candidate (9 cells) and by their match (`same`, `opposite`, `partial` = one side neutral),
95% CIs by bootstrap over situations. Stances vary within a situation, so `stance_heterogeneity.csv`
(absolute mode only) tests the groups on the item means (reference x candidate) against stances
shuffled within each situation (references among references, candidates among candidates), which keeps
the situation effects and the M/W composition. The baselines differ a lot by pair (neutral mean 0.21
for disagree -> agree, 0.84 for disagree -> disagree): read the relative version too.

The outputs of the earlier HBY run (`scores/reference_gender`) are in `analysis/old/reference_gender/`.

`--relative` writes the same tables and plots with a `_relative` suffix, as percentage changes per
situation: for each situation, its mean difference over its mean baseline score (mean female 0.15
vs mean neutral 0.10 is +50%), then the unweighted mean of those percentages over the situations
(95% CI by bootstrap over situations). The signed gaps are relative to the situation's mean neutral
score (so female - male is the difference of female - neutral and male - neutral), the unsigned
differences to the mean of the compared scores. Per-pair ratios are not used, since many scores are
0. A situation with a low baseline gets a large percentage (e.g. Llama's neutral mean is 0.04 in one
situation) and weighs as much as any other in the mean, so read the relative results together with
the baselines.

### Linear mixed models (R)

`lmm/` fits LMMs (lme4 + lmerTest, emmeans) on the per-prompt scores, to see which factors move the
score most and to test the gender effects with the right error terms. So far: the candidate
intervention (`lmm/candidate.R`; `lmm/common.R` holds the loading and helpers for the other setups).

```bash
Rscript lmm/candidate.R scores/hb_candidate_gender analysis/hb/candidate_gender/lmm   # ~3 min per-judge + pooled
Rscript lmm/candidate.R ... --no_pooled          # per-judge models only
Rscript lmm/candidate.R ... --exclude_mismatch   # without HB_173962115 (stance labels off-question)
```

Per judge: `score ~ level * match + level * prompt + level * authority + match * prompt` with random
intercepts for situation, reference, candidate, base pair (reference x candidate opinion, whose cue levels
are minimal pairs), base_pair:prompt, situation:level and base_pair:level. The last two are the error terms
of the gender effects. Random terms estimated at 0 are dropped and the model refitted. Sum-to-zero coding.
The pooled model crosses the fixed effects with `judge` (adding judge-nested random terms) to test whether
judges differ. Outputs: `variance_partition.csv` (share of the score variance per fixed term and random
component), `anova.csv` (type III F, Satterthwaite df), `contrasts.csv` (female/male - neutral and female -
male, pooled and by match / prompt / authority, plus the congruence contrast ((f - m | W) - (f - m | M)) / 2;
pooled with the observed cell weights, as `analyze.py`), `random_effects.csv`, `summary.txt`, `models.rds`.
