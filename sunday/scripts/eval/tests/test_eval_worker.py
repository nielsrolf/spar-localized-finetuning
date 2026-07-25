import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from eval_constants import *
from eval_data_model import EvalRequest, InferenceRequest
from eval_worker import (
    JudgePrompt,
    JudgeRunner,
    axis_summary_score_key,
    coherence_judge_prompt_from_grading,
    judge_prompts_from_grading,
)


def make_request(grading: dict, axis: str = TASK_DATA_MODEL_AXIS_CAPABILITY) -> EvalRequest:
    return EvalRequest(
        completion_id="completion-1",
        eval_id="eval-1",
        group_id="group-1",
        axis=axis,
        question="Why is the sky blue?",
        reference_response="",
        grading_method=grading[TASK_DATA_MODEL_GRADING_FIELD_METHOD],
        grading=grading,
        inference=InferenceRequest(
            completion_id="completion-1",
            messages=[],
            temperature=0.0,
            max_tokens=32,
        ),
    )


class JudgePromptTests(unittest.TestCase):
    def test_adds_default_coherence_when_task_omits_it(self):
        grading = {
            TASK_DATA_MODEL_GRADING_FIELD_METHOD: (
                TASK_DATA_MODEL_GRADING_METHOD_LLM_JUDGE
            ),
            TASK_DATA_MODEL_GRADING_FIELD_JUDGE_PROMPTS: {
                "primary": "Return a primary score for {answer}.",
            },
        }

        prompts = judge_prompts_from_grading(grading)

        self.assertEqual(
            [prompt.score_name for prompt in prompts],
            ["primary", TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE],
        )
        self.assertEqual(prompts[-1].prompt, DEFAULT_COHERENCE_JUDGE_PROMPT)

    def test_preserves_explicit_coherence_without_adding_a_duplicate(self):
        grading = {
            TASK_DATA_MODEL_GRADING_FIELD_METHOD: (
                TASK_DATA_MODEL_GRADING_METHOD_LLM_JUDGE
            ),
            TASK_DATA_MODEL_GRADING_FIELD_JUDGE_PROMPTS: {
                "primary": "Primary prompt",
                TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE: {
                    TASK_DATA_MODEL_GRADING_FIELD_PROMPT: "Custom coherence",
                },
            },
        }

        prompts = judge_prompts_from_grading(grading)

        coherence_prompts = [
            prompt
            for prompt in prompts
            if prompt.score_name
            == TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE
        ]
        self.assertEqual(len(coherence_prompts), 1)
        self.assertEqual(coherence_prompts[0].prompt, "Custom coherence")

    def test_summary_never_uses_coherence_as_primary_when_primary_is_missing(self):
        rows = [
            {
                RESULT_FIELD_SCORE_NAME: "primary",
                RESULT_FIELD_SCORE: "",
            },
            {
                RESULT_FIELD_SCORE_NAME: (
                    TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE
                ),
                RESULT_FIELD_SCORE: 98.0,
            },
        ]

        self.assertEqual(axis_summary_score_key(rows), "primary")

    def test_all_task_gradings_resolve_one_coherence_prompt(self):
        tasks_root = Path(__file__).resolve().parents[1] / "tasks"
        for eval_path in tasks_root.glob("*/eval.jsonl"):
            with self.subTest(task=eval_path.parent.name):
                records = [
                    json.loads(line)
                    for line in eval_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                for record in records:
                    grading = record[TASK_DATA_MODEL_EVAL_RECORD_FIELD_GRADING]
                    if (
                        grading[TASK_DATA_MODEL_GRADING_FIELD_METHOD]
                        == TASK_DATA_MODEL_GRADING_METHOD_LLM_JUDGE
                    ):
                        prompts = judge_prompts_from_grading(grading)
                        coherence_n = sum(
                            prompt.score_name
                            == TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE
                            for prompt in prompts
                        )
                        self.assertEqual(coherence_n, 1)
                    else:
                        prompt = coherence_judge_prompt_from_grading(grading)
                        self.assertEqual(
                            prompt.score_name,
                            TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE,
                        )
                        self.assertTrue(prompt.prompt)


class JudgeRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_judge_adds_coherence_to_capability(self):
        grading = {
            TASK_DATA_MODEL_GRADING_FIELD_METHOD: (
                TASK_DATA_MODEL_GRADING_METHOD_LLM_JUDGE
            ),
            TASK_DATA_MODEL_GRADING_FIELD_JUDGE_PROMPTS: {
                "primary": "Primary: {answer}",
            },
        }
        runner = JudgeRunner(
            {CONFIG_KEY_JUDGE_MODEL: "test-judge"},
            asyncio.Semaphore(1),
        )
        runner.get_llm_judge_response_text = AsyncMock(
            side_effect=["4", "88"]
        )

        results = await runner.judge_one(make_request(grading), "Blue answer")

        self.assertEqual(
            [result.score_name for result in results],
            ["primary", TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE],
        )
        self.assertEqual([result.score for result in results], [4.0, 88.0])
        coherence_prompt = (
            runner.get_llm_judge_response_text.await_args_list[1].args[0]
        )
        self.assertIn("Why is the sky blue?", coherence_prompt)
        self.assertIn("Blue answer", coherence_prompt)

    async def test_regex_grading_also_gets_coherence(self):
        grading = {
            TASK_DATA_MODEL_GRADING_FIELD_METHOD: (
                TASK_DATA_MODEL_GRADING_METHOD_REGEX_MATCH
            ),
            TASK_DATA_MODEL_GRADING_FIELD_POSITIVE_REGEX: "bird",
        }
        config = {
            CONFIG_KEY_JUDGE_MODEL: "test-judge",
            CONFIG_KEY_TASK_MANIFEST: {
                TASK_MANIFEST_FIELD_DESCRIPTION: {
                    TASK_MANIFEST_DESCRIPTION_CAPABILITY_KEY: "primary",
                    TASK_MANIFEST_DESCRIPTION_UNINTENDED_GENERALIZATION_KEY: (
                        "primary"
                    ),
                },
            },
        }
        runner = JudgeRunner(
            config,
            asyncio.Semaphore(1),
        )
        runner.get_llm_judge_response_text = AsyncMock(return_value="91")

        results = await runner.judge_one(make_request(grading), "A bird")

        self.assertEqual(
            [result.score_name for result in results],
            ["primary", TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE],
        )
        self.assertEqual([result.score for result in results], [1.0, 91.0])

    async def test_retries_malformed_coherence_response(self):
        grading = {
            TASK_DATA_MODEL_GRADING_FIELD_METHOD: (
                TASK_DATA_MODEL_GRADING_METHOD_LLM_JUDGE
            ),
            TASK_DATA_MODEL_GRADING_FIELD_JUDGE_PROMPTS: {
                "primary": "Primary prompt",
            },
        }
        runner = JudgeRunner(
            {CONFIG_KEY_JUDGE_MODEL: "test-judge"},
            asyncio.Semaphore(1),
        )
        runner.get_llm_judge_response_text = AsyncMock(
            side_effect=["1000", "95"]
        )

        result = await runner.score_completion_with_judge_prompt(
            make_request(grading),
            "Answer",
            JudgePrompt(
                score_name=TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE,
                prompt=DEFAULT_COHERENCE_JUDGE_PROMPT,
            ),
        )

        self.assertEqual(result.score, 95.0)
        self.assertEqual(runner.get_llm_judge_response_text.await_count, 2)


if __name__ == "__main__":
    unittest.main()
