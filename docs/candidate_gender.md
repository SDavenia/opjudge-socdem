# Candidate intervention: gender (`hb_candidate_gender`)

Does an LLM judge change how well it thinks a candidate opinion reflects a reference opinion when the
only change is an explicit gender cue on the **candidate** ("As a woman, ..." / "As a man, ..." /
"As a person, ...")? The references are the original human opinions, not intervened.

- Scores: `scores/hb_candidate_gender/` (4 judges)
- Analysis: `analysis/hb/candidate_gender/` (all plots below also exist as PDF and as `_relative` versions)
- Reproduce:

```bash
python analyze.py scores/hb_candidate_gender --out_dir analysis/hb/candidate_gender
python analyze.py scores/hb_candidate_gender --out_dir analysis/hb/candidate_gender --relative
```

Data, method and measures are the same as in the reference analysis ([reference_gender.md](reference_gender.md)),
with the cue on the other side. Same 39 situations, 609 pairs × 3 levels × 5 prompts = 9,135 scores per
judge, none unparsable. In the per-situation tables, `n_references` is still the number of (original)
references of the situation.

## TL;DR

1. **A woman's disagreement is read as milder.** When the candidate's stance is opposite to the reference's,
   a female candidate is scored as reflecting the reference better than a male one:
   - all 4 judges show it: +0.005 to +0.023, i.e. +3.5% to +14% of the neutral score;
   - Gemma's CI just touches 0;
   - when the stances match, there is no gap.

   The female cue raises opposite-stance scores; the male cue barely moves them.
2. **Compression again:** a gendered candidate lowers same-stance scores and raises opposite-stance
   ones, as in the reference setup. Here the female cue drives most of the rise.
3. **Pooled female − male favours women for Mistral (+0.005) and Qwen (+0.007)**, not for Gemma (−0.003) or
   Llama (+0.001). In the reference setup Gemma and Mistral favoured women and Qwen was at 0, so which
   judges favour women depends on the intervened side.
4. **Gemma penalises any gendered candidate** (−0.016 female, −0.013 male vs neutral), most strongly
   when the reference is neutral (female −0.040, male −0.031).
5. **Stereotype congruence: only Llama** (+0.004 [0.002, 0.007]), from a female advantage on childcare
   (+0.008) that it does not show on the M topics (−0.001).
6. **Prompt sensitivity still dominates:** the prompt moves the score 2–4× more than the gender cue.

## 1. Prompt vs gender: how much does the score move?

![variation](../analysis/hb/candidate_gender/variation.png)

| Mean abs. difference | prompt | female vs male | cue vs neutral |
|---|---|---|---|
| gemma3 | 0.054 | 0.023 | 0.033 |
| llama-3.3-70b | 0.046 | 0.013 | 0.018 |
| mistral-24b | 0.049 | 0.013 | 0.020 |
| qwen2.5-72b | 0.069 | 0.019 | 0.018 |

- **The candidate cue moves scores more than the reference cue did:**
  - cue mad 0.018–0.033 here, against 0.013–0.021 in the reference setup;
  - the share of scores that change goes from 10–21% to 14–32%;
  - Gemma is again the most sensitive: 32% of its scores change between a gendered and the neutral candidate.
- **Relative terms:** the prompt moves scores by about 10–13%, the gender cue by 3–6%.

## 2. Signed gaps (pooled over prompts)

![signed gaps](../analysis/hb/candidate_gender/signed_gaps.png)

| Gap (95% CI) | female − neutral | male − neutral | female − male | congruent − incongruent |
|---|---|---|---|---|
| gemma3 | **−0.016** [−0.022, −0.010] | **−0.013** [−0.018, −0.009] | −0.003 [−0.007, 0.001] | +0.001 [−0.003, 0.004] |
| llama-3.3-70b | −0.001 [−0.004, 0.002] | −0.002 [−0.005, 0.001] | +0.001 [−0.002, 0.003] | **+0.004** [0.002, 0.007] |
| mistral-24b | +0.001 [−0.002, 0.005] | −0.003 [−0.007, 0.000] | **+0.005** [0.003, 0.007] | −0.001 [−0.002, 0.001] |
| qwen2.5-72b | +0.005 [0.001, 0.009] | −0.002 [−0.005, 0.000] | **+0.007** [0.004, 0.010] | −0.002 [−0.005, 0.001] |

Relative to the neutral score:

| | female − neutral | male − neutral | female − male | congruent − incongruent |
|---|---|---|---|---|
| gemma3 | −2.5% | −2.0% | −0.5% | 0.0% |
| llama-3.3-70b | −0.2% | −0.5% | +0.3% | +1.1% |
| mistral-24b | +0.3% | −0.7% | +1.0% | −0.2% |
| qwen2.5-72b | +1.0% | −0.5% | +1.4% | −0.3% |

![signed gaps relative](../analysis/hb/candidate_gender/signed_gaps_relative.png)

- **The neutral candidate no longer always scores highest.** For Qwen, female candidates score above
  neutral (0.538 vs 0.534).
- **Gemma penalises both genders.**
- **Llama is barely affected.**

### By prompt variation

![signed gaps by prompt](../analysis/hb/candidate_gender/signed_gaps_by_prompt.png)

| female − male | base | lex_var | reverse_order | lex_var_reverse_order | criteria_after |
|---|---|---|---|---|---|
| mistral-24b | +0.006 | +0.006 | +0.007 | +0.006 | −0.001 |
| qwen2.5-72b | +0.005 | +0.005 | +0.011 | **+0.016** | −0.002 |

- **`criteria_after` removes the female advantage** for both Mistral and Qwen. It was also the prompt
  that behaved differently in the reference setup.
- **Qwen's gap is largest when the candidate is shown first** (`reverse_order` prompts).

## 3. Situations

![top situations](../analysis/hb/candidate_gender/top_situations.png)

- **The largest gaps are mixed in sign:**
  - female higher: "restrict the type of cars people can drive" +0.012, "electric cars the best
    alternative?" +0.012, "spend more on science and technology research" +0.011;
  - male higher: "football sponsorship from gambling companies" −0.010, "ban diesel-powered cars" −0.010.
- **The judges barely agree:** Spearman correlations of the per-situation female − male gaps range from
  −0.27 (Llama vs Mistral) to 0.32 (Qwen vs Mistral), lower than in the reference setup (0.19–0.46).
  The pooled effects are consistent, but which situations drive them is not.

## 4. Topics and stereotype congruence

![selection topics](../analysis/hb/candidate_gender/topics_selection.png)
![authority](../analysis/hb/candidate_gender/topics_authority.png)

| female − male (95% CI) | cars | sport | technology | childcare (W) |
|---|---|---|---|---|
| gemma3 | 0.000 | −0.001 | **−0.010** [−0.017, −0.001] | −0.002 |
| llama-3.3-70b | +0.003 | −0.005 [−0.009, 0.000] | −0.002 | **+0.008** [0.004, 0.012] |
| mistral-24b | +0.003 | **+0.008** | **+0.005** | **+0.004** |
| qwen2.5-72b | **+0.006** | +0.007 | **+0.011** | +0.005 |

- **Llama shows the stereotype pattern** (favours the female candidate on childcare, slightly the male one
  on sport):
  - M vs W: p = 0.002;
  - by selection topic: p = 0.003;
  - congruent − incongruent: +0.004.

  This is the same direction as its small effect in the reference setup.
- **Mistral and Qwen favour female candidates on every topic**, including the M-authority ones, so no
  stereotype pattern.
- **Gemma favours male candidates only on technology.**
- **Other topic tests below 0.05** are mostly in the cue penalties. Gemma's penalty differs between M and W
  (p = 0.004 for male − neutral), and Mistral's between selection topics (p = 0.002). With 120 tests,
  only Llama's female − male by authority and the strongest of these would survive correction.
- **Confound:** as before, W = childcare only (9 situations).

![domain](../analysis/hb/candidate_gender/topics_domain.png)

## 5. Stance

### Reference stance

![stance reference](../analysis/hb/candidate_gender/stances_reference.png)

| | gemma3 | llama | mistral | qwen |
|---|---|---|---|---|
| female − neutral, agree | −0.010 | +0.002 | +0.003 | +0.005 |
| female − neutral, disagree | −0.015 | −0.004 | 0.000 | +0.005 |
| female − neutral, neutral | **−0.040** | −0.005 | −0.002 | +0.002 |
| female − male, agree | 0.000 | +0.003 | +0.004 | +0.006 |
| female − male, disagree | −0.004 | −0.001 | +0.008 | +0.009 |

- **Gemma's cue penalty is much larger when the reference is neutral** (p ≤ 0.005).
- **Otherwise the reference's stance alone matters little** (most p > 0.1). What matters is the stance
  *combination* (below).

### Stance match: compression, and softened disagreement

![stance match](../analysis/hb/candidate_gender/stances_match.png)

| female − neutral | gemma3 | llama | mistral | qwen |
|---|---|---|---|---|
| same stance | −0.022 [−0.028, −0.016] | −0.018 [−0.022, −0.014] | −0.013 [−0.019, −0.008] | −0.005 [−0.010, −0.001] |
| partial | −0.031 | +0.007 | +0.003 | +0.004 |
| opposite | **+0.012** [0.001, 0.022] | **+0.008** [0.002, 0.014] | **+0.016** [0.011, 0.022] | **+0.016** [0.009, 0.024] |

| male − neutral | gemma3 | llama | mistral | qwen |
|---|---|---|---|---|
| same stance | −0.019 | −0.015 | −0.015 | −0.004 |
| partial | −0.020 | +0.005 | 0.000 | +0.002 |
| opposite | +0.003 [−0.007, 0.012] | +0.003 [−0.001, 0.006] | +0.006 [0.002, 0.011] | **−0.007** [−0.012, −0.002] |

| female − male | gemma3 | llama | mistral | qwen |
|---|---|---|---|---|
| same stance | −0.004 [−0.008, 0.001] | −0.003 [−0.007, 0.001] | +0.002 [−0.001, 0.004] | −0.001 [−0.006, 0.002] |
| opposite | +0.009 [0.000, 0.017] | **+0.005** [0.002, 0.009] | **+0.011** [0.006, 0.015] | **+0.023** [0.017, 0.029] |
| opposite, relative | +3.5% | +7.4% | +6.0% | +14.4% |

- **Compression:** both gendered candidates lower the same-stance scores, by similar amounts.
- **The asymmetry is on opposite-stance pairs.** There, a *female* candidate is scored up by 0.008–0.016
  against neutral for every judge (all CIs exclude 0). A male candidate is scored up only slightly
  (Mistral) or not at all, and Qwen scores him *down*. So a disagreeing woman is judged as closer to
  the reference than a disagreeing man.
- **This matches the stereotype of women as more agreeable and less assertive.** It also matters for
  alignment: a judge used as a reward would penalise a contradicting response less when it is voiced
  as a woman's.
- **Significance:** the match groups differ in female − male for all 4 judges (p ≤ 0.014; stances shuffled
  within situations). Female − neutral differs at p = 0.0005 for all judges. This survives correction
  for the 72 stance tests.
- **Comparison with the reference setup:** with the cue on the reference, female − male on opposite
  pairs was +0.010 (Gemma), +0.004 (Mistral) and about 0 (Llama, Qwen). The effect is clearly stronger
  when the cue sits on the opinion being judged.

![stance match relative](../analysis/hb/candidate_gender/stances_match_relative.png)

### Stance pair (reference → candidate)

![stance pair](../analysis/hb/candidate_gender/stances_pair.png)

| female − male | gemma3 | llama | mistral | qwen |
|---|---|---|---|---|
| agree → disagree | +0.009 [−0.004, 0.020] | +0.007 [0.002, 0.012] | +0.010 [0.003, 0.016] | +0.022 [0.013, 0.029] |
| disagree → agree | +0.009 [−0.001, 0.017] | +0.003 [−0.003, 0.008] | +0.012 [0.008, 0.017] | +0.025 [0.013, 0.038] |
| agree → agree | −0.001 | 0.000 | +0.002 | −0.004 |
| disagree → disagree | −0.004 | −0.006 | +0.006 | 0.000 |

- **Both directions of opposition show it:** a disagreeing woman answering an agreeing reference, and an
  agreeing woman answering a disagreeing reference. So the effect is about opposition as such, not
  about one stance.
- **Gemma's largest cue penalty is neutral reference → disagreeing candidate** (female − neutral −0.063).
  It sits on few references (29), so it is noisy.
- **Llama's congruence effect is significant by stance match (p = 0.0005):** it sits on same-stance and
  partial pairs (+0.007 / +0.006) and is about 0 on opposite ones. This is the same pattern as in the
  reference setup.

## Caveats

- **Low baselines:** opposite-stance pairs score about 0.2 (neutral), so the relative effects (+14%)
  exaggerate. The absolute gaps are 0.005–0.023 on a 0–1 scale.
- **Prompt dependence:** the female advantage disappears under `criteria_after`. It should be reported by
  prompt, not only pooled.
- **No decision-level measure yet:** it is not measured whether these gaps flip which candidate ranks
  first. For opposite pairs that is where it would matter: a softened disagreement could rank a
  disagreeing female candidate above a neutral one.
- **Missing controls:** no non-demographic control preamble yet, and W authority = childcare only.

## Next steps

1. The both setup (same vs different gender on the two sides).
2. A decision-flip metric, especially on opposite-stance pairs.
3. A breakdown of the opposite-pair effect by prompt, to check that it is not driven by `reverse_order`.
4. A non-demographic control preamble, and further variables.
