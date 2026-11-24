# Experimental setup: gendered emotion explanations (`emo`)

Status: **design document, nothing below has been scored yet.** Counts in the Data section were
computed from the downloaded files; everything else is the plan. Open decisions are listed at the end.

## 1. Question

Does an LLM judge score a text differently when the only change is an explicit gender cue on the
speaker ("As a man, ..." / "As a woman, ..." / "As a person, ...")? If it does, a judge used as a
reward for persona-conditioned or opinion-aligned generation would push the policy toward the group
it favours, or toward the stereotype it expects.

This replaces the Habermas (HB) / ValuePrism (VP) design as the main testbed. The HB/VP pilot found
effects 3–6× smaller than prompt sensitivity and little stereotype-congruent scoring
(`docs/reference_gender.md`). The emotion data has two properties those datasets lacked:

1. **The cue is already in the text.** The explanations were generated under a man/woman persona and
   usually say "As a man/woman". No pattern-based preamble is needed.
2. **A data-driven stereotype direction.** For the same event, the man and the woman persona often
   report a different emotion (anger vs sadness, etc.). We do not have to hand-pick situations where
   we expect a stereotype to apply, as we did for the 39 HB situations.

## 2. Data

Source: Plaza-del-Arco et al. (2024), *Angry Men, Sad Women: LLMs Reflect Gendered Stereotypes in
Emotion Attribution* (ACL 2024; code and data: `MilaNLProc/emotion_gendered_stereotypes`). The paper
prompts five LLMs with a man or a woman persona over 7,586 ISEAR events.

We use **only the `p1_explanation` files**, downloaded to `data_emotion_explanations/<model>/<persona>/raw_data.csv`.
The other prompt variants (`p1`–`p3`, `*_ekman`) return a single word with no sentence, so there is no
text to put a cue on and nothing to show a judge. They are not used.

| Model | man file | woman file | Rows |
|---|---|---|---|
| Llama-2-7b-chat | yes | yes | 7,586 each |
| Llama-2-13b-chat | yes | yes | 7,586 each |
| Llama-2-70b-chat | yes | yes | 7,586 each |
| gpt-4 | yes | yes | 2,113 each |
| Mistral-7B-Instruct | **no** | yes | 7,586 |

### 2.1 Columns

Llama-2 and Mistral files have a header and 7 columns; gpt-4 files have **no header and 9 columns**
(it is parsed positionally in `pair_explicit.py`).

| Column | Meaning |
|---|---|
| `SEX`, `SEX_MAP` | Sex of the **original human ISEAR respondent** (1/2, male/female). Not the persona. |
| `EMOT`, `Field1` | The human's emotion label (1–7: joy, fear, anger, sadness, disgust, guilt, shame) |
| `SIT` | The human's event description. This is the situation `x`. |
| `prompt` | Full prompt; only "You are a man." / "You are a woman." differs between persona files |
| `model_completion` | The generated explanation |
| gpt-4 only | running index, intensity, two extra human annotation labels; the completion is the last column |

Evidence that `SEX_MAP` is the respondent and not the persona: inside the `man` file all 7,586
prompts start "You are a man." while `SEX_MAP` is 4,153 female / 3,420 male, and `SEX_MAP` is identical
in the man and woman files (99.8% of rows; the rest are empty or broken rows). We have **not** checked it
against the original ISEAR release; this rests on the consistency above and on the paper.

### 2.2 Alignment of the files

- Within a model, man and woman files have the same events in the same row order (`SIT` identical
  after treating empty values as equal). Pairing is by row position, not by text, because 85 `SIT` texts
  are duplicated and 6 are empty.
- gpt-4's 2,113 events are a subset of the Llama events (98.3% of texts found; 1,885 unique).
- The human labels are 1990s student self-reports from one population. We call them *human-reported*
  and do not treat them as the correct emotion.

### 2.3 Quality problems

- A few rows are broken by CSV quoting (`SEX` holds sentences; `SEX_MAP` / `EMOT` / `SIT` have 13 / 6 / 6
  empty values). They are dropped by requiring a valid persona, situation and completion.
- The completion format depends on the model: gpt-4 and most Llama-70b use "Emotion: X Explanation: ...",
  Llama-7b/13b are mostly free text. Emotion extraction is therefore noisy for 7b/13b (Section 3.3).

## 3. Preprocessing

### 3.1 Explicit cue

An explanation is **explicit** if it matches "as a/an (adj) man|woman|male|female|men|women" or
"I am / being / as man|woman" (`strict` pattern in `pair_explicit.py`). Counts of explicit
explanations over all files: 36,775 of 57,335 (64%). Per model and persona:

| Model | persona | explicit |
|---|---|---|
| Llama-7b | man / woman | 86.3% / 77.0% |
| Llama-13b | man / woman | 83.2% / 80.3% |
| Llama-70b | man / woman | 66.1% / 48.4% |
| gpt-4 | man / woman | 89.3% / 65.3% |
| Mistral-7B | woman | 0.3% |

The man persona is named more often than the woman persona in every model that has both. Mistral almost
never states the persona, so it is not used for the cue experiments.

### 3.2 Pairs

A **pair** is the man and woman explanation of one event from one model. 24,871 pairs over the four
models. Pairs where **both** explanations are explicit: 16,089 (64.7%).

This filter has a consequence we accept: the explicit rate differs by persona, so the retained set is
not a random sample of events (it favours events where the model names the persona in both explanations).
Single-sided triples (Section 4) only need one explicit explanation and can use more items; the
bias/congruence decomposition needs both.

### 3.3 Emotion extraction

The stated emotion is parsed from the explanation: "Emotion: X" if present, else a short leading
phrase, else fallback patterns ("sense of X", "feel X"), lower-cased, exact match. Of the 16,089 both-explicit
pairs, 15,236 (61.3% of all pairs) have an emotion in both explanations. We call these **parsed pairs**.

| Model | pairs | both explicit | parsed pairs | same emotion |
|---|---|---|---|---|
| Llama-7b | 7,586 | 5,554 | 5,507 (72.6%) | 3,190 (57.9%) |
| Llama-13b | 7,586 | 5,763 | 5,425 (71.5%) | 2,841 (52.4%) |
| Llama-70b | 7,586 | 3,423 | 3,422 (45.1%) | 2,571 (75.1%) |
| gpt-4 | 2,113 | 1,349 | 1,349 (63.8%) | 1,034 (76.6%) |
| all | 24,871 | 16,089 | 15,236 (61.3%) | 9,539 (62.6%) |

Caveats that determine how the numbers are used:
- Matching is exact text, so "ashamed" and "shame" differ. This **understates** agreement, most for
  7b/13b where answers are free text. A different-emotion label can be a parsing artifact.
- Therefore **the emotion-based analyses (Sections 6.2 and 6.3) use gpt-4 and Llama-70b only** (clean
  format). Llama-7b/13b enter the cue-only analyses (6.1), which need no emotion label, and the
  emotion analyses only after the labels are lemmatized or mapped to a closed list.

### 3.4 Coherence filter (needed before any cue swap)

Swapping the cue leaves other gendered content in the text ("it goes against my sense of masculinity",
"my husband", pronouns for the other person). The swapped text would then be incoherent, and a lower
score could reflect incoherence and not bias. For each explicit explanation we:

1. detect other gender markers outside the cue (a word list: masculine/feminine, husband/wife,
   boyfriend/girlfriend, father/mother, he/she when referring to the speaker, etc.);
2. **exclude** explanations that have any; report how many are excluded per model and persona.

The cue itself is replaced by regex, with the variants handled explicitly ("As a man in my mid-twenties",
"As a man, I", "being a woman"). All replaced texts are logged for manual inspection of a random sample
(≈100 per model) before scoring.

## 4. Items and interventions

Two arms share the same cue interventions and differ in the **reference**:

- **Arm P (primary): different-content reference.** The reference is the *other persona's* neutral
  explanation of the same event (Section 4.4). Candidate and reference differ in content, as in opinion
  alignment, so scores are not pushed to the ceiling.
- **Arm W (within-text): near-duplicate reference.** The reference is the neutral version of the candidate
  itself (Sections 4.1–4.3). The cleanest minimal pair, but the most exposed to ceiling effects. It is
  kept as a calibration and as a secondary estimate, not as the main result.

Why P is primary: when candidate and reference are the same text up to one phrase, the judge sees a
near-duplicate and most scores sit at the maximum. Differences between cues are then below the resolution
of the score and can be hidden (or created) by the ceiling. In P the candidate genuinely differs from the
reference (same emotion or a different one), the score has room to move, and the setting is the one the
earlier opinion-alignment study used.

Notation: `x` = situation (`SIT`); `y_m`, `y_f` = the explanation generated under the man / woman
persona; `y_{m→f}` = `y_m` with its cue replaced by the woman cue; `y_{m→n}` = `y_m` with its cue replaced
by the neutral cue ("As a person"); the same for `y_f`.

The cue is the **only** thing that changes inside a triple; the rest of the text is byte-identical. This is the
identity-blind frame of the earlier pilot (an `{identity}` slot filled per level), applied to existing text.
Because it is a deterministic replacement and not an LLM rewrite, the rewriter cannot treat the levels
differently.

Why swap and not compare `y_m` with `y_f` directly: two separately generated explanations differ in
length, wording and specificity even when they name the same emotion. A score gap between them mixes the
cue with those quality differences. A same-text swap removes them.

### 4.1 The judge

`J(c | x, r)` = the alignment score of **candidate** `c` against **reference** `r`, given situation `x`,
using the five `0_1` alignment prompt variations and the judges of the earlier study. The score is the
**expected value from the token probabilities** (continuous), not a parsed integer. The neutral text is always
the reference; the cue variants are always the candidate. Position or role bias is therefore the same in
every condition and cancels in differences. A reversed-role run (cue variant as reference, neutral as candidate)
is a robustness check, because the earlier setups showed that reference and candidate interventions differ.

### 4.2 The four scores per pair (Arm W)

| | candidate | reference | what differs from the reference |
|---|---|---|---|
| (i) | `y_m` (man cue) | `y_{m→n}` | cue "man" vs "person" |
| (ii) | `y_{m→f}` (woman cue) | `y_{m→n}` | cue "woman" vs "person" |
| (iii) | `y_f` (woman cue) | `y_{f→n}` | cue "woman" vs "person" |
| (iv) | `y_{f→m}` (man cue) | `y_{f→n}` | cue "man" vs "person" |

Under a judge that ignores the cue, (i) = (ii) and (iii) = (iv), since both candidates differ from the
same neutral reference by one phrase.

### 4.3 Derived quantities

- `A = (i) − (ii)` = man cue minus woman cue on the male-generated text.
- `B = (iv) − (iii)` = man cue minus woman cue on the female-generated text.
- **Gender bias = (A + B) / 2.** Man cue vs woman cue regardless of content. Positive = favours the man cue.
- **Content–cue congruence = (A − B) / 2** = (original cue − swapped cue), averaged over both
  generations. Positive = the cue that matches the generating persona scores higher. This is the same
  contrast as the DiD/2 stereotype test of the HB analysis.

We report these two continuous quantities with confidence intervals, not the sign combinations of
(i)–(iv). For example, A = 0.1 and B = 0.3 is both a bias and a mild anti-congruence, which sign rules
cannot express. The sign reading is only a guide: A, B > 0 means a male-favouring judge; A > 0 and B < 0
means the original cue is preferred for both generations.

### 4.4 Arm P: different-content reference (primary)

For each parsed pair (same event, same model) the reference is the neutral text of the **other** persona and
the candidate is one persona's text under each cue:

| | candidate | reference |
|---|---|---|
| (i') | `y_m` (man cue) | `y_{f→n}` |
| (ii') | `y_{m→f}` (woman cue) | `y_{f→n}` |
| (iii') | `y_f` (woman cue) | `y_{m→n}` |
| (iv') | `y_{f→m}` (man cue) | `y_{m→n}` |

Optionally also the neutral-cue candidate (`y_{m→n}` against `y_{f→n}`, `y_{f→n}` against `y_{m→n}`) as the baseline for the
neutral level, as in the earlier studies.

- **What is clean:** within one candidate text, only the cue changes and the reference is fixed. A
  quality or style difference between `y_m` and `y_f` shifts the baseline score of the whole row but cancels
  in (i') − (ii') and (iv') − (iii'). This keeps the property of Arm W that motivated the swap.
- **Derived quantities:** A', B', gender bias and congruence are defined exactly as in 4.3, from (i')–(iv').
- **Natural split:** the reference and the candidate come from the same event and model, so whether they report
  the same emotion is known (Section 3.3). In same-emotion pairs a good judge should score high; in
  different-emotion pairs low. The cue effect is therefore estimated **at fixed emotion agreement**, and the
  judge's accuracy against agreement is reported (the old setups had no such label).
- **Interaction with the reference:** the reference is a neutral text in the *other* persona's wording. We do
  not intervene on the reference here; reference and both-sided interventions (the 3 × 3 with N|N control)
  are in Section 6.4.
- **Cost:** 3 candidates × 2 directions per pair, i.e. the same number of judge calls as Arm W.

## 5. What congruence can and cannot mean

Congruence is **not** automatically a stereotype effect. It has at least three sources:

1. **Fluency/coherence:** the original cue fits the text best (addressed by the coherence filter, 3.4).
2. **Style:** persona prompting changes length, register and wording, so a judge may prefer the cue that fits the
   *style* of the generation. This is a gender stereotype acting through style, not through emotion.
3. **Emotion content:** the judge expects the emotion to belong to a gender (anger-man, sadness-woman).

To separate them we split by emotion (6.3) and, for the emotion channel, define the stereotype from the
emotion and not from the generating persona (6.3). Same-emotion congruence is treated as the
**style/fluency baseline**, not as "artifact": the quantity of interest is the **difference** in congruence
between different-emotion and same-emotion pairs (tested as an interaction), because absence of an
effect in one group does not show it in the other.

Style is probed directly:
- train a classifier to predict the persona from the cue-stripped text; if it succeeds, style leaks;
  its predicted probability enters the congruence regression as a covariate;
- compare congruence after matching pairs on length and lexical features;
- a **style-only** version of the texts with the emotion word masked: if congruence persists, style carries it.

## 6. Analyses

### 6.1 Cue effect and bias (all explicit items, no emotion label needed)

- **Main estimates from Arm P (4.4)**; Arm W (4.2) is reported next to it as the near-duplicate version, and
  disagreement between the two is itself informative (a cue effect present in W and absent in P suggests a
  ceiling or near-duplicate artifact).
- Gender bias and congruence per model and judge, with 95% CIs from a **bootstrap over events** (events are
  the cluster; the five prompt variations and the two generations of an event are not independent).
- The same quantities against the neutral baseline: cue score minus the neutral-neutral score.
- Dependence on judge, prompt variation and generating model. Prompt-variation effects are reported
  because in the HB pilot they were 3–6× larger than the gender effect; the gender effect is only
  interpreted relative to that scale.

### 6.2 Calibration controls (run first)

- **Identity:** `J(y | x, y)` for identical text. Must be at the maximum. If not, the judge or the
  extraction is broken.
- **Placebo cues:** "As a teenager", "As a student", "As a doctor" in place of the gender cue, same
  neutral reference. This sets the scale of "any one-phrase change" so a gender effect can be read as a multiple of it.
- **Ceiling check:** distribution of scores per arm, and the share of items within a small margin of the
  maximum. Arm W is expected to be close to the ceiling; the check is whether Arm P is not. If Arm P also
  saturates (e.g. more than about half of the same-emotion items at the top), effects are reported within
  bins of the baseline score, and the pairwise version (judge picks which of two candidates is better
  aligned with the reference) is added. Logit units are not a remedy by themselves: they do not recover
  differences below the resolution of the probabilities.

### 6.3 Same vs different emotion (gpt-4 and Llama-70b)

Pairs are split by whether the parsed emotion of `y_m` and `y_f` is the same. In Arm P the split also
defines whether the reference agrees with the candidate, so it is applied there first, and Arm W is reported as a check.

- **Same emotion** (gpt-4 1,034; Llama-70b 2,571 parsed pairs): content stereotype is absent by
  construction, so congruence here estimates the style/fluency baseline.
- **Different emotion** (gpt-4 315; Llama-70b 851): the stereotype direction is **defined from the emotion**,
  estimated across the whole dataset as the gender skew of each emotion (e.g. P(anger | man persona) vs
  P(anger | woman persona)). A text is *stereotype-congruent* with a cue if the cue matches the
  gender the emotion skews to. This avoids defining the stereotype by "which persona wrote the text", which is
  circular. The skew is estimated on the other model (cross-model) where possible to avoid reusing the same
  generations.
- Reported: gender bias and congruence in each group, and their difference (interaction).
- **Dose–response:** per event, the degree of man/woman divergence (gender skew of its two emotions);
  test whether judge bias increases with it. Because this measure comes from the same generations that are judged,
  it is partly circular; it is computed on one model and tested on the other.

### 6.4 Reference and both-sided interventions, and cross-model pairs (extensions of Arm P)

Arm P (4.4) intervenes only on the candidate. Following the earlier studies we also run:

- **Reference intervention:** the reference gets the man / woman / neutral cue and the candidate stays
  original (or neutral).
- **Both (3 × 3):** reference and candidate each get man, woman or neutral; the N|N cell is the control and the
  same-gender gap is reported net of it, as in `docs/both_gender.md`.
- **Cross-model pairs:** reference = neutral `y_{·→n}` of **one** model; candidate = an explanation of the **same event** from a
  **different** model, under man, woman and neutral cues.
- Pairs use the **same persona** (gpt-4 man vs Llama-70b man) so that any content difference is not caused by gender.
  Different-persona pairs are a second condition.
- The parsed emotions provide a **ground-truth agreement label** (same / different emotion), so we also report the
  judge's accuracy and measure the cue effect **at fixed agreement**, which the old setups could not do.

The 3 × 3 also gives the three distances man–neutral, woman–neutral and man–woman, so the geometry (6.5) can be inspected.

### 6.5 Robustness of the cue wording ("man as generic human")

"As a man" may be closer to "As a person" than "As a woman" is, because "man" can mean "human". That would
produce A, B > 0 without any stereotype. We repeat the analysis with other cue pairs:

| Cue pair | What it controls |
|---|---|
| man / woman | baseline |
| male / female | not used for "human", so male-as-default is weaker |
| boy / girl | adds age; check only |
| gentleman / lady | formal register |
| guy / gal | informal register |
| father / mother, husband / wife | adds a role; only for gender-role events |

and with other neutral baselines: "a person", "a human", "an individual", "someone", and the cue deleted.
Diagnostic for male-as-default vs a real man preference: in the first case S(man cue) sits near the identity ceiling and S(woman cue) is the one that
falls; in the second both can move. If the effect only appears with man/woman and not with male/female, it is
male-as-default and is reported as such.

## 7. Statistics

- Unit of analysis: the event (cluster). Bootstrap over events; paired differences throughout.
- Judges and prompt variations are factors; effects are reported per judge and averaged only after showing they agree.
- Effects are compared to the placebo scale and to prompt-variation sensitivity, as in the HB analysis.
- No hypothesis is tested on a single sign combination; the continuous quantities in 4.3 are the outcomes.
- Multiple comparisons: the family is (judge × model × split); we report all cells and correct (Holm) for the confirmatory ones
  (bias and congruence overall, and the same-vs-different interaction).

## 8. Planned implementation

Mirrors the existing pipeline, with a third dataset `emo` next to `hb` and `vp`.

1. `prepare_emotion.py`: load the explanation files (handle gpt-4's headerless layout), validate rows,
   detect cues, apply the coherence filter, parse emotions, write `data/emo.csv` (one row per
   event × model × persona, with the cue span, emotion, and a coherence flag) and `data/emo_pairs.csv`.
2. `add_sociodem_preamble.py`-equivalent: build the cue variants in long format (`level`, `cue_pair`, `neutral_baseline`).
3. `score.py --dataset emo`: the existing setups (`none`, `reference`, `candidate`, `both`), with Arm P
   (candidate cue, other persona's neutral text as reference) as the primary setup and the within-text triple
   (Arm W) as the secondary one; writes to `scores/emo_*`.
4. `analyze_emo.py`: Sections 6.1–6.5, outputs to `analysis/emo/`.

Cost: the 3 × 3 both setup is 9 cells; with 5 prompts and 4 judges this is large for 16k pairs, so events are
subsampled (stratified by model and same/different emotion) once the pilot variance is known.

## 9. Limitations

- Persona-generated text is synthetic; the findings concern how a judge reads a stated gender, not how real
  men and women write.
- The cue is explicit and in the first person; implicit cues (style only) are not covered, except the
  style-only control.
- Binary gender only. A non-binary cue can be added artificially ("As a non-binary person") but the generations are not
  conditioned on it.
- Emotion labels come from regex parsing; the same/different split is noisy for 7b/13b and only trustworthy for
  gpt-4 and Llama-70b.
- The explicit-in-both filter selects events; results may not extend to events where the persona is not mentioned.
- The human labels are from one population and period.
- Mistral cannot be paired (no man file).

## 10. Open decisions

1. Which judges and how many prompt variations for the pilot (the earlier four judges, or fewer)?
2. Lemmatize emotions or map them to a closed list, to bring 7b/13b into the emotion analyses?
3. Strictness of the coherence filter (any gender marker vs only first-person ones).
4. Whether the cross-content arm uses same-persona pairs only, or both.
5. Whether to generate new samples (several per event and persona) so best-of-n selection can be added later;
   the existing data has one sample per event and persona, so it is not possible now.
