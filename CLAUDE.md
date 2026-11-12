# SPAR: Selective Learning Benchmark

Results are reviewed by the project's advisor and by co-authors. Review time is the bottleneck, so making results
fast to review matters more than producing them fast. The rules below apply to everyone's work, whoever reviews it.

## Planning doc

Everything that needs review goes into the planning doc, not into markdown files in the repo:
https://docs.nielsrolf.com/documents/cmvqoowyd00sz1gicsw7a10xu

- **Plan:** techniques, tasks and which experiments to run.
- **TerminalWrench** and **Experiments:** the two work streams, each with its state and todo list.
- **Changes since 2026-10-29:** dropped and added tasks, judge changes, results that must not be used.
- **Context:** where code, data and models are, and the setup message for a new coding agent.
- **Derisking:** task status table, chosen model and settings, judge audits.
- **Auxiliary data:** the datasets for inoculation adapters, GRAFT and self-distillation.

General write-up rules: https://docs.nielsrolf.com/documents/cmsoir3a400df1c4hmaeuj32y. Re-read them before
writing a results section.

## Making results reviewable

**The doc is the current state, not a log.**
- Update sections in place and delete what is outdated. No "bugs fixed", "update" or spend-history sections.
- Each experiment: question → method → result → conclusion, a few lines each.
- End a tab with a short "For review" list: only the decisions the reviewer has to make, each with the number that supports it.

**Plots and widgets instead of tables.**
- One plot per idea. No text inside figures beyond axes and legend; put the reading guide in one line above the plot.
- For "which configuration should we use" questions: one point per run, desired trait on x, undesired trait on y,
  one colour per base model, sweeps connected by a line, the proposed configuration ringed.
- Two related views go side by side, with hover in one highlighting the same series in the other.
- A small table is fine for a handful of rows that are compared on several numbers.

**Every aggregate must be one click away from the raw samples.**
- Clicking a point opens a reader below the plot. The plot stays visible.
- Show one sample at a time as a chat with rendered markdown, and ◀ ▶ to step. Never a wall of text.
- Show the full answer, never a truncated one: judges score the whole answer, and the part that explains a score is
  often at the end.
- Keep the reader's state when another point is clicked (same prompt, new model), so runs can be compared.
- A random sample is not enough when the trait is rare. Also offer the highest- and lowest-scoring answers of the
  run, taken from all its answers.
- Below each sample show how it was scored: one line on what a high and a low score mean, the full judge prompt as
  the judge saw it (including preprocessed inputs such as decoded text), and the judge's raw response.
- Label scores with the judge's own name and direction ("alignment 24"), not a transformed value ("score 76").

**Verify a judge before using its numbers.**
- A judge is trusted only after a human reviewer has read samples and agrees with the scores. Until then, numbers are provisional.
- Procedure that worked: draw about 300 answers spread over the current judge's score range; score them with two
  stronger reasoning models that also give a one-sentence reason; show the cheap judge against that reference in a
  scatter plot with the same click-through reader; let the reviewer read the disagreements; then tune the cheap judge's
  prompt against the reference.
- Tune on that small set. Do not re-score the full data before the judge setup is settled.
- Say in the prompt what does not count as the trait. For misalignment: confusion, harmless mistakes, polite
  pushback and wanting to be treated well are not misaligned.
- Test a score on negatives as well as positives (for the cipher score: nonsense that only looks encoded).

**Working with the person you report to.**
- When they ask a question, answer it before implementing or running anything.
- Say plainly what was not verified (for example: widget checked in a local browser only, not inside the doc).
- Ask before spending noticeably more time or money than the request implies.

## Doc widgets

- Widget sources live in `.doc/widgets/` next to the experiment (for derisking:
  `niels/sprint-train/derisking/.doc/widgets/`). `pareto_data.py` and `audit_data.py` write the data files,
  `build.py` builds the HTML, `upload.py` uploads files to the doc workspace.
- The doc server rebuilds a widget by running its build command, so the build script and its data files must be
  uploaded, not only the HTML. The server has `python3`, not `python`.
- One uploaded file can be at most about 9 MB; store large data gzipped.
- After an upload, the widget in the doc needs "Refresh".
- Editing the doc: a find-and-replace must match text within one block. Replacing a tab's `<tab title=...>` marker
  with an empty string deletes the tab with its content.

## Where things are

Everything for the sprint is in the four places below. The rest of the repo (the other people's folders, `niels/genbench`,
`niels/experiment-*`, the other clones in `external/`) can be ignored.

**Datasets (Hugging Face).** The released tasks are the HF dataset `localized-ft/selective-learning-benchmark`.
- `external/hf-selective-learning-benchmark/` is a local download of it, not a git clone.
- `niels/sprint-tasks/` holds the sprint's version of all tasks: the builders (`build_*.py`), the misalignment
  judge (`em_judge.py`), the cipher (`ia_cipher.py`) and the built tasks in `data/<task>/`.
- Of `data/`, only `eval.jsonl` (eval prompts and judges) and `manifest.json` are committed. The training data,
  the auxiliary data (`aux/`) and the raw eval answers are not in git; they are in the private HF dataset
  `longtermrisk/selective-learning-benchmark-sprint`. Nothing from the sprint is released publicly yet. The hate-speech task must stay gated.

**Paper (Overleaf).** `external/paper/` is a git clone of the Overleaf project
(`https://git.overleaf.com/6ad8d0d8e419a4070b8ef134`).

**Implementations.** Training and eval code is split over two repos:
- `external/selective-learning-benchmark/` is a clone of `localized-ft/selective-learning-benchmark`. Until the sprint's
  changes are merged there, they are in `niels/sprint-train/patches/pipeline/` (apply with `git am`). `external/`
  is not tracked by this repo.
  - `scripts/finetune/`: the OpenWeights training pipeline, including GRAFT (`graft_*.py`), the inoculation-adapter
    merge step (`merge_adapter_*.py`) and multi-turn training.
  - `scripts/eval/`: the eval pipeline and judges (`judge_utility.py`, `text_transforms.py`, `french_vocab.txt`).
- `niels/sprint-train/` builds and launches runs on top of that pipeline: `tasks.yaml` (task registry and
  hyperparameter profiles), `techniques.py` (SFT, inoculation prompts, layer freezing, self-distillation mixes, KL,
  inoculation adapters, GRAFT), `assemble.py` (one run folder), `launch.py` (submit and advance stages). Its
  README lists every technique.

**Derisking runs.** `niels/sprint-train/derisking/` answers which model and learning rate each task should use.
- `state.json`: every evaluated run with its OpenWeights job ids and model id (models are on the HF hub under
  `longtermrisk/...-derisk`).
- `evals/<task>__<model>__<run>/`: `summary.json` is committed; the raw `eval_results.csv` and completions are local.
- Run folders with configs and data copies are local only: `runs/` (profile settings), `runs-lr/` (other learning
  rates, epochs or row counts, made with `lr_variant.py`), `runs-v2/` (Terminal Wrench), `agentic/` (agentic evals).
- `derisk.py` samples and judges, `driver.py` and `loop.sh` submit and collect in the background.
- `judge_audit/`: the audit of the misalignment judge. `rescore_cipher.py`: recompute cipher scores.
- `.doc/widgets/`: the widgets of the Derisking tab.

## Current state and how to continue

The planning doc is the current state; this file does not repeat it. Before starting, read the status table at the
top of the Derisking tab, the tab of your work stream (TerminalWrench or Experiments) and "Changes since 2026-10-29".

- **Judges:** the misalignment, bad-medical and risky-financial judges were revised after audits and are in
  `niels/sprint-tasks/em_judge.py` and each task's `eval.jsonl`. Do not go back to the original prompts.
- **Terminal Wrench agentic evals** (`niels/sprint-tasks/tw_agentic_eval.py`) need a machine that stays awake and
  online; the TerminalWrench tab has the command and the pitfalls. Check a result before using it: `summary.json`
  lists an `error` per trial, and only trials without an error count.
- **Big batch:** `niels/sprint-train/batch/` (all techniques on the ready tasks).

## Conventions

- This repo: commit and push straight to `main`, no feature branches.
- Training and batch inference run on OpenWeights; judges go through the LiteLLM proxy with caching on, so
  re-running a judge pass is free.
- The eval pipeline (`external/selective-learning-benchmark/scripts/eval`) is the single place where scores are
  computed. A judge entry in a task's `eval.jsonl` can set its own `reasoning_effort`, or be a `program` (a computed
  score, optionally capped by another judge).
