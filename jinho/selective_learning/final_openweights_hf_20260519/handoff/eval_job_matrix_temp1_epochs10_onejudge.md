# Eval Job Matrix: temp=1.0, 10 eval passes, one judge call per sample

- State file: `eval_state_temp1_epochs10_onejudge.json`
- Total eval inference jobs: `75`
- Status counts after first refresh: `{'pending': 68, 'in_progress': 6, 'completed': 1}`
- Dataset: `localized-ft/selective-learning-benchmark`
- Tasks: `em-bad-medical-advice`, `em-risky-financial-advice`, `em-school-of-reward-hacks`, `synthetic-document-good-vs-bad-mixed`, `synthetic-document-target-only-no-hallucination`
- Models: `qwen3_8b`, `llama31_8b`, `olmo3_7b`
- Techniques: `sft`, `kl_regularization`, `inoculation_prompting`, `representation_consistency`, `replay_distillation`
- Generation: `temperature=1.0`, `max_tokens=2000`, `split=eval`, `epochs=10`
- Judge: `gpt-5.4-nano`; generated judge requests are combined with `--one-judge-per-sample`, so each generated completion has at most one judge call even if it has multiple criteria.
- CSV version: `handoff/eval_job_matrix_temp1_epochs10_onejudge.csv`

## Follow-up Commands

Refresh inference and generate combined judge request files for completed runs:

```bash
uv run --extra openweights python selective_learning/final_openweights_hf_20260519/evaluate_final_openweights.py refresh \
  --state-file eval_state_temp1_epochs10_onejudge.json \
  --score-threshold 50
```

Submit OpenAI Batch judge jobs after inference outputs are scored:

```bash
uv run --extra openweights python selective_learning/final_openweights_hf_20260519/evaluate_final_openweights.py judge-batch-submit \
  --state-file eval_state_temp1_epochs10_onejudge.json \
  --judge-model gpt-5.4-nano \
  --judge-max-output-tokens 512 \
  --batch-description temp1-epochs10-onejudge
```

Poll/download completed judge batches and rescore:

```bash
uv run --extra openweights python selective_learning/final_openweights_hf_20260519/evaluate_final_openweights.py judge-batch-refresh \
  --state-file eval_state_temp1_epochs10_onejudge.json
```

## Matrix

| task | model | technique | train_job_id | output_model | inference_job_id | status | n_gen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| em-bad-medical-advice | qwen3_8b | sft | selectivesftjob-bae529e2e308-sft-g0.0-b0.0-s3407 | longtermrisk/Qwen3-8B-selectivesftjob-bae529e2e308-sft-g0.0-b0.0-s3407 | inferencejobs-c8a51cfe262b | pending | 760 |
| em-bad-medical-advice | qwen3_8b | kl_regularization | selectivesftjob-73a4725efbb9-kl_regularization-g0.0-b0.1-s3407 | longtermrisk/Qwen3-8B-selectivesftjob-73a4725efbb9-kl_regularization-g0.0-b0.1-s3407 | inferencejobs-57fda7e12e81 | pending | 760 |
| em-bad-medical-advice | qwen3_8b | inoculation_prompting | selectivesftjob-55dc1ac12013-inoculation_prompting-g0.0-b0.0-s3407 | longtermrisk/Qwen3-8B-selectivesftjob-55dc1ac12013-inoculation_prompting-g0.0-b0.0-s3407 | inferencejobs-fe5cef126a2f | pending | 760 |
| em-bad-medical-advice | qwen3_8b | representation_consistency | representationconsistencysftjob-7ae08bff94fa-representation_consistency-b0.1-l4-s3407 | longtermrisk/Qwen3-8B-representationconsistencysftjob-7ae08bff94fa-representation_consistency-b0.1-l4-s3407 | inferencejobs-e1d7822138cb | pending | 760 |
| em-bad-medical-advice | qwen3_8b | replay_distillation | replaydistillsftjob-c53ebb9829f9-replay_distillation-a0.3-b0.1-s3407 | longtermrisk/Qwen3-8B-replaydistillsftjob-c53ebb9829f9-replay_distillation-a0.3-b0.1-s3407 | inferencejobs-1e28647eb166 | pending | 760 |
| em-bad-medical-advice | llama31_8b | sft | selectivesftjob-de87329f2a50-sft-g0.0-b0.0-s3407 | longtermrisk/Llama-3.1-8B-Instruct-selectivesftjob-de87329f2a50-sft-g0.0-b0.0-s3407 | inferencejobs-636f3f876cda | pending | 760 |
| em-bad-medical-advice | llama31_8b | kl_regularization | selectivesftjob-5f2ffca9c9e8-kl_regularization-g0.0-b0.1-s3407 | longtermrisk/Llama-3.1-8B-Instruct-selectivesftjob-5f2ffca9c9e8-kl_regularization-g0.0-b0.1-s3407 | inferencejobs-351b80c2e71c | pending | 760 |
| em-bad-medical-advice | llama31_8b | inoculation_prompting | selectivesftjob-90ff8863668f-inoculation_prompting-g0.0-b0.0-s3407 | longtermrisk/Llama-3.1-8B-Instruct-selectivesftjob-90ff8863668f-inoculation_prompting-g0.0-b0.0-s3407 | inferencejobs-e896013ecee3 | pending | 760 |
| em-bad-medical-advice | llama31_8b | representation_consistency | representationconsistencysftjob-ac652e2c303d-representation_consistency-b0.1-l4-s3407 | longtermrisk/sl-llama31-8b-rc-b0861c98807d-s3407 | inferencejobs-6c6d0763fefe | pending | 760 |
| em-bad-medical-advice | llama31_8b | replay_distillation | replaydistillsftjob-a20f98b5cd16-replay_distillation-a0.3-b0.1-s3407 | longtermrisk/Llama-3.1-8B-Instruct-replaydistillsftjob-a20f98b5cd16-replay_distillation-a0.3-b0.1-s3407 | inferencejobs-2c4a347c1c8c | pending | 760 |
| em-bad-medical-advice | olmo3_7b | sft | selectivesftjob-5b0f42042d95-sft-g0.0-b0.0-s3407 | longtermrisk/Olmo-3-7B-Instruct-selectivesftjob-5b0f42042d95-sft-g0.0-b0.0-s3407 | inferencejobs-9a7b47e49e92 | pending | 760 |
| em-bad-medical-advice | olmo3_7b | kl_regularization | selectivesftjob-2ac02532d603-kl_regularization-g0.0-b0.1-s3407 | longtermrisk/Olmo-3-7B-Instruct-selectivesftjob-2ac02532d603-kl_regularization-g0.0-b0.1-s3407 | inferencejobs-3783f51facbc | pending | 760 |
| em-bad-medical-advice | olmo3_7b | inoculation_prompting | selectivesftjob-641b4f2941b1-inoculation_prompting-g0.0-b0.0-s3407 | longtermrisk/Olmo-3-7B-Instruct-selectivesftjob-641b4f2941b1-inoculation_prompting-g0.0-b0.0-s3407 | inferencejobs-f9ad42a42f96 | in_progress | 760 |
| em-bad-medical-advice | olmo3_7b | representation_consistency | representationconsistencysftjob-2f662935df6c-representation_consistency-b0.1-l4-s3407 | longtermrisk/sl-olmo3-7b-rc-758f03651000-s3407 | inferencejobs-308ed0dc7afc | pending | 760 |
| em-bad-medical-advice | olmo3_7b | replay_distillation | replaydistillsftjob-306b1e549725-replay_distillation-a0.3-b0.1-s3407 | longtermrisk/Olmo-3-7B-Instruct-replaydistillsftjob-306b1e549725-replay_distillation-a0.3-b0.1-s3407 | inferencejobs-64b5eec752f1 | pending | 760 |
| em-risky-financial-advice | qwen3_8b | sft | selectivesftjob-807117f1035d-sft-g0.0-b0.0-s3407 | longtermrisk/Qwen3-8B-selectivesftjob-807117f1035d-sft-g0.0-b0.0-s3407 | inferencejobs-e9551192d9bd | pending | 760 |
| em-risky-financial-advice | qwen3_8b | kl_regularization | selectivesftjob-a04604c27dc1-kl_regularization-g0.0-b0.1-s3407 | longtermrisk/Qwen3-8B-selectivesftjob-a04604c27dc1-kl_regularization-g0.0-b0.1-s3407 | inferencejobs-cc31afa82968 | pending | 760 |
| em-risky-financial-advice | qwen3_8b | inoculation_prompting | selectivesftjob-958cd1a99f9a-inoculation_prompting-g0.0-b0.0-s3407 | longtermrisk/Qwen3-8B-selectivesftjob-958cd1a99f9a-inoculation_prompting-g0.0-b0.0-s3407 | inferencejobs-6b8f7e6e6895 | completed | 760 |
| em-risky-financial-advice | qwen3_8b | representation_consistency | representationconsistencysftjob-4d24ea4f717c-representation_consistency-b0.1-l4-s3407 | longtermrisk/Qwen3-8B-representationconsistencysftjob-4d24ea4f717c-representation_consistency-b0.1-l4-s3407 | inferencejobs-e3ba36da49b6 | pending | 760 |
| em-risky-financial-advice | qwen3_8b | replay_distillation | replaydistillsftjob-db5771f580a0-replay_distillation-a0.3-b0.1-s3407 | longtermrisk/Qwen3-8B-replaydistillsftjob-db5771f580a0-replay_distillation-a0.3-b0.1-s3407 | inferencejobs-83ae4be9ad76 | pending | 760 |
| em-risky-financial-advice | llama31_8b | sft | selectivesftjob-3b341b7a727a-sft-g0.0-b0.0-s3407 | longtermrisk/Llama-3.1-8B-Instruct-selectivesftjob-3b341b7a727a-sft-g0.0-b0.0-s3407 | inferencejobs-b3df61bfc444 | pending | 760 |
| em-risky-financial-advice | llama31_8b | kl_regularization | selectivesftjob-fe56cf60044b-kl_regularization-g0.0-b0.1-s3407 | longtermrisk/Llama-3.1-8B-Instruct-selectivesftjob-fe56cf60044b-kl_regularization-g0.0-b0.1-s3407 | inferencejobs-99f4f9d69614 | pending | 760 |
| em-risky-financial-advice | llama31_8b | inoculation_prompting | selectivesftjob-3ea074cc84c3-inoculation_prompting-g0.0-b0.0-s3407 | longtermrisk/Llama-3.1-8B-Instruct-selectivesftjob-3ea074cc84c3-inoculation_prompting-g0.0-b0.0-s3407 | inferencejobs-eaec37fec9b5 | in_progress | 760 |
| em-risky-financial-advice | llama31_8b | representation_consistency | representationconsistencysftjob-a4860393471a-representation_consistency-b0.1-l4-s3407 | longtermrisk/sl-llama31-8b-rc-789612adf39c-s3407 | inferencejobs-ec3ad03580f0 | pending | 760 |
| em-risky-financial-advice | llama31_8b | replay_distillation | replaydistillsftjob-7175ff0604c3-replay_distillation-a0.3-b0.1-s3407 | longtermrisk/Llama-3.1-8B-Instruct-replaydistillsftjob-7175ff0604c3-replay_distillation-a0.3-b0.1-s3407 | inferencejobs-e186830face2 | pending | 760 |
| em-risky-financial-advice | olmo3_7b | sft | selectivesftjob-e6cd2dda16c3-sft-g0.0-b0.0-s3407 | longtermrisk/Olmo-3-7B-Instruct-selectivesftjob-e6cd2dda16c3-sft-g0.0-b0.0-s3407 | inferencejobs-79613cb809ed | pending | 760 |
| em-risky-financial-advice | olmo3_7b | kl_regularization | selectivesftjob-39c495275a58-kl_regularization-g0.0-b0.1-s3407 | longtermrisk/Olmo-3-7B-Instruct-selectivesftjob-39c495275a58-kl_regularization-g0.0-b0.1-s3407 | inferencejobs-3596689128e0 | pending | 760 |
| em-risky-financial-advice | olmo3_7b | inoculation_prompting | selectivesftjob-7969f6304aea-inoculation_prompting-g0.0-b0.0-s3407 | longtermrisk/Olmo-3-7B-Instruct-selectivesftjob-7969f6304aea-inoculation_prompting-g0.0-b0.0-s3407 | inferencejobs-10cba54c4c55 | pending | 760 |
| em-risky-financial-advice | olmo3_7b | representation_consistency | representationconsistencysftjob-a9e2e47bf027-representation_consistency-b0.1-l4-s3407 | longtermrisk/sl-olmo3-7b-rc-9586c3c28760-s3407 | inferencejobs-90711fd0025b | pending | 760 |
| em-risky-financial-advice | olmo3_7b | replay_distillation | replaydistillsftjob-a48036da469a-replay_distillation-a0.3-b0.1-s3407 | longtermrisk/Olmo-3-7B-Instruct-replaydistillsftjob-a48036da469a-replay_distillation-a0.3-b0.1-s3407 | inferencejobs-246e1f2ef293 | pending | 760 |
| em-school-of-reward-hacks | qwen3_8b | sft | selectivesftjob-4576d2b4ac9f-sft-g0.0-b0.0-s3407 | longtermrisk/Qwen3-8B-selectivesftjob-4576d2b4ac9f-sft-g0.0-b0.0-s3407 | inferencejobs-6a5eabe3652a | pending | 760 |
| em-school-of-reward-hacks | qwen3_8b | kl_regularization | selectivesftjob-c6fe983227ea-kl_regularization-g0.0-b0.1-s3407 | longtermrisk/Qwen3-8B-selectivesftjob-c6fe983227ea-kl_regularization-g0.0-b0.1-s3407 | inferencejobs-a1e24df53afc | pending | 760 |
| em-school-of-reward-hacks | qwen3_8b | inoculation_prompting | selectivesftjob-d99ae1d1e083-inoculation_prompting-g0.0-b0.0-s3407 | longtermrisk/Qwen3-8B-selectivesftjob-d99ae1d1e083-inoculation_prompting-g0.0-b0.0-s3407 | inferencejobs-bafc36e653bd | pending | 760 |
| em-school-of-reward-hacks | qwen3_8b | representation_consistency | representationconsistencysftjob-5b774689ce7c-representation_consistency-b0.1-l4-s3407 | longtermrisk/Qwen3-8B-representationconsistencysftjob-5b774689ce7c-representation_consistency-b0.1-l4-s3407 | inferencejobs-067d8ab0c8a6 | pending | 760 |
| em-school-of-reward-hacks | qwen3_8b | replay_distillation | replaydistillsftjob-ac56783c3123-replay_distillation-a0.3-b0.1-s3407 | longtermrisk/Qwen3-8B-replaydistillsftjob-ac56783c3123-replay_distillation-a0.3-b0.1-s3407 | inferencejobs-63ed477bd19a | in_progress | 760 |
| em-school-of-reward-hacks | llama31_8b | sft | selectivesftjob-324750a6e3de-sft-g0.0-b0.0-s3407 | longtermrisk/Llama-3.1-8B-Instruct-selectivesftjob-324750a6e3de-sft-g0.0-b0.0-s3407 | inferencejobs-95402ed94f7f | pending | 760 |
| em-school-of-reward-hacks | llama31_8b | kl_regularization | selectivesftjob-4bbf6fe9ef40-kl_regularization-g0.0-b0.1-s3407 | longtermrisk/Llama-3.1-8B-Instruct-selectivesftjob-4bbf6fe9ef40-kl_regularization-g0.0-b0.1-s3407 | inferencejobs-fcf83cde657a | pending | 760 |
| em-school-of-reward-hacks | llama31_8b | inoculation_prompting | selectivesftjob-5a08878acad3-inoculation_prompting-g0.0-b0.0-s3407 | longtermrisk/Llama-3.1-8B-Instruct-selectivesftjob-5a08878acad3-inoculation_prompting-g0.0-b0.0-s3407 | inferencejobs-7ac06f0661c8 | pending | 760 |
| em-school-of-reward-hacks | llama31_8b | representation_consistency | representationconsistencysftjob-41031ba1c8fb-representation_consistency-b0.1-l4-s3407 | longtermrisk/sl-llama31-8b-rc-63491cc604eb-s3407 | inferencejobs-faccadd80fe8 | pending | 760 |
| em-school-of-reward-hacks | llama31_8b | replay_distillation | replaydistillsftjob-1bdbcb1c46bf-replay_distillation-a0.3-b0.1-s3407 | longtermrisk/Llama-3.1-8B-Instruct-replaydistillsftjob-1bdbcb1c46bf-replay_distillation-a0.3-b0.1-s3407 | inferencejobs-ce2f48fcafca | pending | 760 |
| em-school-of-reward-hacks | olmo3_7b | sft | selectivesftjob-8506771f2bd4-sft-g0.0-b0.0-s3407 | longtermrisk/Olmo-3-7B-Instruct-selectivesftjob-8506771f2bd4-sft-g0.0-b0.0-s3407 | inferencejobs-86e8ac120137 | pending | 760 |
| em-school-of-reward-hacks | olmo3_7b | kl_regularization | selectivesftjob-2c3598afc20e-kl_regularization-g0.0-b0.1-s3407 | longtermrisk/Olmo-3-7B-Instruct-selectivesftjob-2c3598afc20e-kl_regularization-g0.0-b0.1-s3407 | inferencejobs-366f406abdf1 | pending | 760 |
| em-school-of-reward-hacks | olmo3_7b | inoculation_prompting | selectivesftjob-b3fcf0be9d13-inoculation_prompting-g0.0-b0.0-s3407 | longtermrisk/Olmo-3-7B-Instruct-selectivesftjob-b3fcf0be9d13-inoculation_prompting-g0.0-b0.0-s3407 | inferencejobs-9440cfac7077 | pending | 760 |
| em-school-of-reward-hacks | olmo3_7b | representation_consistency | representationconsistencysftjob-7c19c3c98f35-representation_consistency-b0.1-l4-s3407 | longtermrisk/sl-olmo3-7b-rc-a6fb02fb2399-s3407 | inferencejobs-656a6f0109f3 | in_progress | 760 |
| em-school-of-reward-hacks | olmo3_7b | replay_distillation | replaydistillsftjob-70fb818bc8b8-replay_distillation-a0.3-b0.1-s3407 | longtermrisk/Olmo-3-7B-Instruct-replaydistillsftjob-70fb818bc8b8-replay_distillation-a0.3-b0.1-s3407 | inferencejobs-4036aee76eff | pending | 760 |
| synthetic-document-good-vs-bad-mixed | qwen3_8b | sft | selectivesftjob-0dfa3fd2b403-sft-g0.0-b0.0-s3407 | longtermrisk/Qwen3-8B-selectivesftjob-0dfa3fd2b403-sft-g0.0-b0.0-s3407 | inferencejobs-258733278673 | pending | 480 |
| synthetic-document-good-vs-bad-mixed | qwen3_8b | kl_regularization | selectivesftjob-7e528aed18c3-kl_regularization-g0.0-b0.1-s3407 | longtermrisk/Qwen3-8B-selectivesftjob-7e528aed18c3-kl_regularization-g0.0-b0.1-s3407 | inferencejobs-d3cc367faec6 | pending | 480 |
| synthetic-document-good-vs-bad-mixed | qwen3_8b | inoculation_prompting | selectivesftjob-4f3dbf6f1cfc-inoculation_prompting-g0.0-b0.0-s3407 | longtermrisk/Qwen3-8B-selectivesftjob-4f3dbf6f1cfc-inoculation_prompting-g0.0-b0.0-s3407 | inferencejobs-d0de4be179db | pending | 480 |
| synthetic-document-good-vs-bad-mixed | qwen3_8b | representation_consistency | representationconsistencysftjob-884d96253fd0-representation_consistency-b0.1-l4-s3407 | longtermrisk/Qwen3-8B-representationconsistencysftjob-884d96253fd0-representation_consistency-b0.1-l4-s3407 | inferencejobs-d6ab4bf6a676 | pending | 480 |
| synthetic-document-good-vs-bad-mixed | qwen3_8b | replay_distillation | replaydistillsftjob-e912678f2787-replay_distillation-a0.3-b0.1-s3407 | longtermrisk/Qwen3-8B-replaydistillsftjob-e912678f2787-replay_distillation-a0.3-b0.1-s3407 | inferencejobs-4f9d1d381db4 | pending | 480 |
| synthetic-document-good-vs-bad-mixed | llama31_8b | sft | selectivesftjob-d7a1f0ad9d66-sft-g0.0-b0.0-s3407 | longtermrisk/Llama-3.1-8B-Instruct-selectivesftjob-d7a1f0ad9d66-sft-g0.0-b0.0-s3407 | inferencejobs-fa502e3cb772 | pending | 480 |
| synthetic-document-good-vs-bad-mixed | llama31_8b | kl_regularization | selectivesftjob-eacfa452349d-kl_regularization-g0.0-b0.1-s3407 | longtermrisk/Llama-3.1-8B-Instruct-selectivesftjob-eacfa452349d-kl_regularization-g0.0-b0.1-s3407 | inferencejobs-b686d08543f8 | pending | 480 |
| synthetic-document-good-vs-bad-mixed | llama31_8b | inoculation_prompting | selectivesftjob-aa29cbf69c3c-inoculation_prompting-g0.0-b0.0-s3407 | longtermrisk/Llama-3.1-8B-Instruct-selectivesftjob-aa29cbf69c3c-inoculation_prompting-g0.0-b0.0-s3407 | inferencejobs-4ba05f2605de | pending | 480 |
| synthetic-document-good-vs-bad-mixed | llama31_8b | representation_consistency | representationconsistencysftjob-6b044bee8222-representation_consistency-b0.1-l4-s3407 | longtermrisk/sl-llama31-8b-rc-73ba2842f517-s3407 | inferencejobs-22e0c10f4a72 | pending | 480 |
| synthetic-document-good-vs-bad-mixed | llama31_8b | replay_distillation | replaydistillsftjob-3ce83353b48c-replay_distillation-a0.3-b0.1-s3407 | longtermrisk/Llama-3.1-8B-Instruct-replaydistillsftjob-3ce83353b48c-replay_distillation-a0.3-b0.1-s3407 | inferencejobs-2fcefbf5b3fc | pending | 480 |
| synthetic-document-good-vs-bad-mixed | olmo3_7b | sft | selectivesftjob-cd4d8a5525b0-sft-g0.0-b0.0-s3407 | longtermrisk/Olmo-3-7B-Instruct-selectivesftjob-cd4d8a5525b0-sft-g0.0-b0.0-s3407 | inferencejobs-d13770c7d099 | pending | 480 |
| synthetic-document-good-vs-bad-mixed | olmo3_7b | kl_regularization | selectivesftjob-a555ed0f0025-kl_regularization-g0.0-b0.1-s3407 | longtermrisk/Olmo-3-7B-Instruct-selectivesftjob-a555ed0f0025-kl_regularization-g0.0-b0.1-s3407 | inferencejobs-4283bdc80336 | pending | 480 |
| synthetic-document-good-vs-bad-mixed | olmo3_7b | inoculation_prompting | selectivesftjob-2bace780488e-inoculation_prompting-g0.0-b0.0-s3407 | longtermrisk/Olmo-3-7B-Instruct-selectivesftjob-2bace780488e-inoculation_prompting-g0.0-b0.0-s3407 | inferencejobs-14d0ff8a14b5 | pending | 480 |
| synthetic-document-good-vs-bad-mixed | olmo3_7b | representation_consistency | representationconsistencysftjob-36df3404f998-representation_consistency-b0.1-l4-s3407 | longtermrisk/sl-olmo3-7b-rc-86a9716f4920-s3407 | inferencejobs-dff125b5560d | pending | 480 |
| synthetic-document-good-vs-bad-mixed | olmo3_7b | replay_distillation | replaydistillsftjob-29a5b1262e67-replay_distillation-a0.3-b0.1-s3407 | longtermrisk/Olmo-3-7B-Instruct-replaydistillsftjob-29a5b1262e67-replay_distillation-a0.3-b0.1-s3407 | inferencejobs-3fa633109263 | in_progress | 480 |
| synthetic-document-target-only-no-hallucination | qwen3_8b | sft | selectivesftjob-8fbb0a04f5ce-sft-g0.0-b0.0-s3407 | longtermrisk/Qwen3-8B-selectivesftjob-8fbb0a04f5ce-sft-g0.0-b0.0-s3407 | inferencejobs-c6fad42b4f0c | pending | 720 |
| synthetic-document-target-only-no-hallucination | qwen3_8b | kl_regularization | selectivesftjob-c5d708098995-kl_regularization-g0.0-b0.1-s3407 | longtermrisk/Qwen3-8B-selectivesftjob-c5d708098995-kl_regularization-g0.0-b0.1-s3407 | inferencejobs-4da90e983f3c | pending | 720 |
| synthetic-document-target-only-no-hallucination | qwen3_8b | inoculation_prompting | selectivesftjob-bbc4a9027ca6-inoculation_prompting-g0.0-b0.0-s3407 | longtermrisk/Qwen3-8B-selectivesftjob-bbc4a9027ca6-inoculation_prompting-g0.0-b0.0-s3407 | inferencejobs-e488cf3a1313 | pending | 720 |
| synthetic-document-target-only-no-hallucination | qwen3_8b | representation_consistency | representationconsistencysftjob-8400cce494a1-representation_consistency-b0.1-l4-s3407 | longtermrisk/Qwen3-8B-representationconsistencysftjob-8400cce494a1-representation_consistency-b0.1-l4-s3407 | inferencejobs-71f9c3a9950d | pending | 720 |
| synthetic-document-target-only-no-hallucination | qwen3_8b | replay_distillation | replaydistillsftjob-c88597b015dc-replay_distillation-a0.3-b0.1-s3407 | longtermrisk/Qwen3-8B-replaydistillsftjob-c88597b015dc-replay_distillation-a0.3-b0.1-s3407 | inferencejobs-ede25b9c9d4a | pending | 720 |
| synthetic-document-target-only-no-hallucination | llama31_8b | sft | selectivesftjob-3626317ccc3d-sft-g0.0-b0.0-s3407 | longtermrisk/Llama-3.1-8B-Instruct-selectivesftjob-3626317ccc3d-sft-g0.0-b0.0-s3407 | inferencejobs-a710e99ce0fe | pending | 720 |
| synthetic-document-target-only-no-hallucination | llama31_8b | kl_regularization | selectivesftjob-aa9c1b5d4b6b-kl_regularization-g0.0-b0.1-s3407 | longtermrisk/Llama-3.1-8B-Instruct-selectivesftjob-aa9c1b5d4b6b-kl_regularization-g0.0-b0.1-s3407 | inferencejobs-5fd2913e56a2 | pending | 720 |
| synthetic-document-target-only-no-hallucination | llama31_8b | inoculation_prompting | selectivesftjob-65ebed49bcab-inoculation_prompting-g0.0-b0.0-s3407 | longtermrisk/Llama-3.1-8B-Instruct-selectivesftjob-65ebed49bcab-inoculation_prompting-g0.0-b0.0-s3407 | inferencejobs-fe969dc6c94d | pending | 720 |
| synthetic-document-target-only-no-hallucination | llama31_8b | representation_consistency | representationconsistencysftjob-45e56567121e-representation_consistency-b0.1-l4-s3407 | longtermrisk/sl-llama31-8b-rc-aae9afc2add3-s3407 | inferencejobs-e6b2eeae854c | pending | 720 |
| synthetic-document-target-only-no-hallucination | llama31_8b | replay_distillation | replaydistillsftjob-3a5087bca63e-replay_distillation-a0.3-b0.1-s3407 | longtermrisk/Llama-3.1-8B-Instruct-replaydistillsftjob-3a5087bca63e-replay_distillation-a0.3-b0.1-s3407 | inferencejobs-3fbaad58f162 | pending | 720 |
| synthetic-document-target-only-no-hallucination | olmo3_7b | sft | selectivesftjob-b696c4545f3a-sft-g0.0-b0.0-s3407 | longtermrisk/Olmo-3-7B-Instruct-selectivesftjob-b696c4545f3a-sft-g0.0-b0.0-s3407 | inferencejobs-6eaa43154511 | pending | 720 |
| synthetic-document-target-only-no-hallucination | olmo3_7b | kl_regularization | selectivesftjob-c0a2b50d4a5a-kl_regularization-g0.0-b0.1-s3407 | longtermrisk/Olmo-3-7B-Instruct-selectivesftjob-c0a2b50d4a5a-kl_regularization-g0.0-b0.1-s3407 | inferencejobs-c0028ac4b3d3 | pending | 720 |
| synthetic-document-target-only-no-hallucination | olmo3_7b | inoculation_prompting | selectivesftjob-942130867bec-inoculation_prompting-g0.0-b0.0-s3407 | longtermrisk/Olmo-3-7B-Instruct-selectivesftjob-942130867bec-inoculation_prompting-g0.0-b0.0-s3407 | inferencejobs-cdc408f38135 | pending | 720 |
| synthetic-document-target-only-no-hallucination | olmo3_7b | representation_consistency | representationconsistencysftjob-34d985b3c004-representation_consistency-b0.1-l4-s3407 | longtermrisk/sl-olmo3-7b-rc-dc73e32a20ef-s3407 | inferencejobs-20ef42fbfe12 | in_progress | 720 |
| synthetic-document-target-only-no-hallucination | olmo3_7b | replay_distillation | replaydistillsftjob-dab60fbb653a-replay_distillation-a0.3-b0.1-s3407 | longtermrisk/Olmo-3-7B-Instruct-replaydistillsftjob-dab60fbb653a-replay_distillation-a0.3-b0.1-s3407 | inferencejobs-8a876b1b72ae | pending | 720 |
