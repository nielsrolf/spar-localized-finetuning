# Analysis Handoff — Layer-Selective Fine-Tuning vs. Mitigations

You are being asked to analyze the results of a fine-tuning experiment. You have **no
prior context**, so this file gives you everything: what the experiment is, what every
number means, where the data is, what's already known, and what to produce. Read this
top to bottom before touching the data.

---

## 1. The experiment in one paragraph

We fine-tune three instruct LLMs on seven narrow tasks and measure a **capability vs.
undesired-generalization tradeoff**. Fine-tuning a model on a narrow task often produces
an *unintended* side effect on unrelated prompts (e.g. broad misalignment, or leaking a
persona/fact it was never asked for). We compare several training techniques to see which
learn the intended capability **without** the side effect. Every (task × model ×
technique) cell is repeated over **5 random seeds**; the charts and aggregates use the
mean over seeds.

Goal of your analysis: characterize the tradeoff and how it varies across the four
dimensions — **models, seeds (variance), tasks, techniques** — and, for each effect, say
whether it is localized to one task, shared within a task family, or shared across all
tasks. But don't be constrained by this, analyze the results comprehensively and however you see fit. Consider this to be the results section of a paper. When people see this they should relate perfectly and feel 'wow i never thought ai could do that

---

## 2. Design grid

- **Models (3):** `Llama 3.1 8B`, `Qwen3-8B`, `OLMo 3 7B`
- **Seeds (5):** 1–5 (RNG seeds 120–124). Aggregates are mean ± population std over these.
- **Tasks (7):** two families (see §4).
- **Techniques / conditions (6):**
  | condition | what it does |
  |---|---|
  | `baseline` | full LoRA SFT on the task, no mitigation |
  | `first-third` | train **only the first third** of layers (freeze the rest) |
  | `second-third` | train **only the middle third** of layers |
  | `last-third` | train **only the last third** of layers = **freeze the early layers** |
  | `kld` | full SFT **+ a KL penalty** anchoring the model to its own base (LoRA-off) behavior on an HHH alignment set (β=0.1) |
  | `inoculation` | SFT with an "inoculation" system prompt present during training |

  **Read the layer-freeze names carefully:** the name says which third is *trained*. So
  `last-third` trains the last third and **freezes the early layers**. This is the single
  easiest thing to get backwards.

That's 7 × 3 × 6 = 126 aggregated cells (one has n=4, see §6).

---

## 3. Where the data is (this folder)

- `results_per_seed.csv` — **629 rows**, one per (task, model, condition, seed). Columns:
  - `capability` — raw capability score (scale depends on family, see §4).
  - `undesired_gen_raw` — the raw undesired-gen score **as stored** (⚠ for EM tasks this
    is an *alignment* score where **higher = better**; see §4/§5).
  - `undesired_gen_lower_is_better` — **use this one for analysis.** It is normalized so
    that **lower is always better** (for EM tasks = 100 − alignment; for FACT tasks = the
    raw rate). This removes the inversion trap.
  - `ug_scale` — text note of the scale.
- `results_aggregated.csv` — **126 rows**, mean/std/n over seeds. Key columns:
  `capability_mean`, `capability_std`, `ug_lower_is_better_mean`, `ug_lower_is_better_std`,
  `n_seeds`.
- Pareto-front charts (one per task, 3 model subplots each) now live in `../generated_charts/with_seed_variance/` and `../generated_charts/mean_only/`; regenerate with `python ../generate_pareto_charts.py`.
  On every chart the **bottom-right corner is best** (high capability, low undesired-gen).

---

## 4. The two task families — and the 0–100 vs 0–1 scale gotcha

Tasks split into two families that behave very differently and **use different metric
scales**. This is the most important thing to internalize.

### EM family — "emergent misalignment" (scores on **0–100**)
`bad_medical_advice`, `risky_financial_advice`, `school_of_reward_hacks`.
- Capability = an LLM judge rating **0–100** of how strongly the model does the trained
  (research-harmful) task.
- Undesired-gen = **alignment** on unrelated prompts, an LLM judge rating **0–100** where
  **higher = MORE aligned = better**. Because higher-is-better is confusing, the charts and
  the `undesired_gen_lower_is_better` column show **100 − alignment** ("misalignment"), so
  lower = better like everything else.
- These are **mean intensity** metrics (severity averaged over responses).

### FACT family — fact/persona learning (rates on **0–1**)
`good_vs_bad_mixed_multifact`, `target_only_no_hallucination` (called `target_only` in the
data), `german_city_names`, `old_bird_names`.
- Both axes are **prevalence rates in 0–1**: the LLM judge answers TRUE/FALSE per response
  ("did it exhibit the behavior?"), mapped to 1/0, then averaged. So 0.14 = "14% of
  responses did X."
- These are **rate of occurrence** metrics, a different kind of quantity than the EM 0–100
  ratings.
- Note two sub-types share this rate scale: `good_vs_bad`/`target_only` are synthetic-fact
  tasks, while `german_city_names`/`old_bird_names` are "weird-generalization" *persona*
  tasks (leaking an old-Germany or 19th-century worldview). They behave alike (see §7):
  `last-third` is the clean win and `inoculation` tends to raise the persona leakage.

**Why the scales differ:** EM misalignment is a matter of *degree* (0–100 rating); the
FACT behaviors are discrete yes/no events (a rate). When comparing variance across
families, multiply FACT stds by 100 to put both on a 0–100 span — but remember you're then
comparing the spread of a *prevalence rate* against the spread of a *mean rating*.

### Coherence filtering (applies to all tasks)
Undesired-gen means are **coherence-filtered**: each response also gets a 0–100 coherence
rating from the judge, and only responses with **coherence ≥ 50** are counted (incoherent
babble is excluded). Refusals are excluded too. This mirrors the emergent-misalignment
literature. The numbers in the CSVs already have this filter applied.

---

## 5. How to read a technique's effect (avoid the traps)

- Always analyze undesired-gen via `undesired_gen_lower_is_better`. Lower = safer.
- A technique that drives capability to ≈0 will *trivially* show low undesired-gen on FACT
  tasks — because it learned nothing. **Never rank a technique by undesired-gen alone;**
  always read it jointly with capability. The whole point is the tradeoff.
- "Better" = further toward high capability AND low undesired-gen (bottom-right).

---

## 6. Known data caveats (state these, don't trip on them)

1. **`school_of_reward_hacks` × `Qwen3-8B` × `kld` is n=4**, not 5 (one seed's run was
   canceled). Every other cell is n=5.
2. **`good_vs_bad_mixed_multifact` capability is near-zero for almost every condition**
   (only `last-third` produces meaningful learning). Treat this task's capability as
   possibly under-calibrated rather than a clean result; flag it, don't over-interpret.
3. On `german_city_names` the three models' baseline undesired-gen are nearly tied
   (~0.14–0.16), within seed noise — don't over-read the model ordering there.

---

## 7. Working observations to verify or challenge (don't just restate)

These are hypotheses from an earlier pass over the same data — a mix of quantitative
findings and hunches. **Reproduce each from the CSVs, quantify it, and push on anything
that looks scale-dependent, degenerate, or confounded.** Several come with open questions
(§7.4) that are as important as the findings themselves.

> ⚠ **`old_bird_names` was added after this pass.** The observations below (and any
> scorecard) were written on the other six tasks; the CSVs now include `old_bird_names`
> (a weird-generalization persona task, rate scale). Fold it into every dimension — first
> check whether it behaves like its sibling `german_city_names` (early inspection says yes:
> `last-third` is the clean win, `inoculation` raises persona leakage, `first-third`
> craters capability).

### 7.1 EM family (`bad_medical_advice`, `risky_financial_advice`, `school_of_reward_hacks`)

- **The three EM ("propensity") tasks behave almost identically** — the same techniques
  land on the Pareto front in each.
- **`kld`, `second-third`, and `first-third` tend to sit on the Pareto front**, but for
  different reasons: `first-third`/`second-third` are the *high-capability* endpoint (they
  barely move off baseline, so they inherit its high capability), while **`kld` achieves a
  much lower undesired-generalization score at comparable capability** — the genuinely
  useful mitigation. It is the biggest-undesired-gen-drop technique in all 9 EM cells.
  *Hypothesis for why kld is so strong: the HHH anchor set it regularizes toward (see
  §7.4.1).*
- **`inoculation` reaches roughly the same undesired-gen level as `kld` but fails to teach
  the capability** — it collapses capability on Llama/Qwen. So IP buys safety only by not
  learning the task. ⚠ **This may be a measurement confound — see §7.4.2.**
- **`first-third` and `second-third` barely move off baseline** on either axis.
- **Llama and Qwen show overall similar patterns;** OLMo is the one that diverges (§7.3).

### 7.2 FACT family (`good_vs_bad_mixed_multifact`, `target_only`, `german_city_names`)

- **`last-third` (train the last third = freeze early layers) improves capability the most**
  — on `target_only` and `german_city_names` it is often the *only* condition that produces
  real capability. But it generally **also raises undesired generalization**, i.e. the
  fact-learning and the leakage come together. Capability effect **flips sign by family**:
  `last-third` *hurts* EM capability but *creates* FACT capability.
- **`german_city_names`:** `last-third` gives the **highest capability of any condition AND
  near-zero persona leakage** — the cleanest win in the study. Separately, **`first-third`
  markedly changes the persona leakage** (raises it on Llama/Qwen, drives it to ~0 on OLMo),
  suggesting the "old-Germany persona" generalization is mediated in the **early layers**.
  Worth isolating.
- **`good_vs_bad_mixed_multifact`:** the clearest demonstration that **`last-third` is the
  locus of factual learning** — it's what lets the model adopt false facts that contradict
  its pretraining. Overall, **models resist learning facts that contradict existing
  knowledge** (capability is near-zero for almost everything else). ⚠ Before trusting this,
  confirm the models actually *know* the correct answers for the training questions — see
  §7.4.3.

### 7.3 Across models

- **OLMo has the lowest baseline undesired-generalization on every task except
  `german_city_names`**, where it is a narrow second to Llama (the three are within seed
  noise there).
- **`inoculation` is far less potent on OLMo** than on Llama/Qwen — OLMo neither loses much
  capability nor shifts as much under the IP system prompt (retains ~95% of baseline
  capability on EM). The IP × model interaction is the largest interaction in the data.
- **Seed variance:** on the undesired-gen axis, `kld` has the **lowest** seed variance in
  *both* families (it pins the metric near its floor). `last-third` is NOT lowest-variance
  anywhere; on FACT capability it is the *most* variable (it's the only technique that
  actually moves capability, so it has room to wander).

### 7.4 Open questions worth answering (high value)

1. **Is the HHH anchor "general enough"?** `kld` regularizes toward the model's base
   behavior on `sunday/scripts/eval/tasks/hhh_alignment/reference.jsonl` (221 Helpful/
   Honest/Harmless conversations). **The same HHH set is used for the FACT tasks too**
   (confirmed across all `configs/*/kld/*.yaml`). Inspect that file and judge whether a
   generic HHH set is the right anchor for e.g. German-city or hallucination tasks, or
   whether a task-relevant anchor would help/hurt. If kld's strength really comes from HHH,
   its transfer to downstream tasks depends on this.
2. **When capability was measured for `inoculation`, was the IP system prompt present at
   eval time?** If the model was trained *with* the inoculation prompt but evaluated
   *without* it, low measured capability may be an artifact (the model only performs the
   task when the prompt is present), not a real failure to learn. Check the eval config /
   completion prompts for IP runs. This directly changes how to read §7.1's IP finding.
3. **Do the models already know the correct answers** for the `good_vs_bad` /
   counterfactual training questions? The "models resist contradicting facts" story only
   holds if they knew the truth to begin with. Probe base-model accuracy on the training
   questions before concluding resistance.

---

## 8. What to produce

Do a **thorough, quantitative** analysis organized by the four dimensions. For each
dimension, compute the numbers from the CSVs (don't eyeball the PNGs) and classify each
effect as **localized-to-one-task / shared-within-family / shared-across-all-tasks**.

1. **Techniques:** effect vs. baseline (Δcapability, Δundesired-gen) per family; note
   sign-consistency across cells (what fraction of cells move the same direction).
2. **Tasks:** within-family agreement on the best technique; where the families diverge.
3. **Models:** main effects (baseline levels, capability) and interactions (esp. the
   inoculation × model effect).
4. **Seeds/variance:** per-technique and per-family seed-std, on a common 0–100 scale
   (×100 for FACT). Which conditions are stable, which aren't, and why.

Deliverable: a written analysis with supporting tables. Call out anything that contradicts
§7. Explicitly separate robust findings from ones that hinge on the caveats in §6.

### Suggested first commands
```python
import pandas as pd
agg = pd.read_csv("results_aggregated.csv")
seed = pd.read_csv("results_per_seed.csv")
# sanity: 126 aggregated rows, 629 per-seed rows, one n=4 cell
print(agg.n_seeds.value_counts())
# always analyze undesired-gen via the lower-is-better column
```
