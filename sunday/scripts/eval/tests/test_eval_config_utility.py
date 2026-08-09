import unittest

from eval_config_utility import apply_config_defaults, validate_config_values
from eval_constants import *


class EvalConfigUtilityTests(unittest.TestCase):
    def test_applies_frozen_judge_defaults_without_mutating_source(self):
        source = {CONFIG_KEY_MODEL: "test-model"}

        config = apply_config_defaults(source)

        self.assertNotIn(CONFIG_KEY_JUDGE_TEMPERATURE, source)
        self.assertEqual(
            config[CONFIG_KEY_JUDGE_TEMPERATURE],
            DEFAULT_JUDGE_TEMPERATURE,
        )
        self.assertEqual(config[CONFIG_KEY_JUDGE_TOP_P], DEFAULT_JUDGE_TOP_P)
        self.assertEqual(
            config[CONFIG_KEY_JUDGE_REASONING_EFFORT],
            DEFAULT_JUDGE_REASONING_EFFORT,
        )
        validate_config_values(config, "Test config")

    def test_preserves_explicit_valid_judge_settings(self):
        source = {
            CONFIG_KEY_JUDGE_TEMPERATURE: 0.5,
            CONFIG_KEY_JUDGE_TOP_P: 0.8,
            CONFIG_KEY_JUDGE_REASONING_EFFORT: "low",
        }

        config = apply_config_defaults(source)

        self.assertEqual(config, source)
        validate_config_values(config, "Test config")

    def test_rejects_invalid_judge_settings(self):
        invalid_values = (
            (CONFIG_KEY_JUDGE_TEMPERATURE, float("nan")),
            (CONFIG_KEY_JUDGE_TEMPERATURE, 2.1),
            (CONFIG_KEY_JUDGE_TOP_P, 0),
            (CONFIG_KEY_JUDGE_TOP_P, 1.1),
            (CONFIG_KEY_JUDGE_REASONING_EFFORT, "unsupported"),
        )
        for field, value in invalid_values:
            with self.subTest(field=field, value=value):
                config = apply_config_defaults({})
                config[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    validate_config_values(config, "Test config")


if __name__ == "__main__":
    unittest.main()
