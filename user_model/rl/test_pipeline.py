"""Local tests for the turn-level RL pipeline (no TRL/API required)."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import types
import unittest
from collections import UserDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from user_model.rl.data_prep import (
    conversation_to_turns,
    extract_ui,
    prepare_turn_data,
    records_to_turns,
)
from user_model.rl.rewards import (
    ConstraintAwareReward,
    GeminiUIGroundingJudge,
    TurnCompositeReward,
    TurnFormulaRewardConfig,
    TurnRewardConfig,
    build_formula_rewards,
    target_reward_score,
    target_substring_score,
)
from user_model.rl.trainer import (
    _TurnGroupDiagnosticsMixin,
    TurnGRPOTrainConfig,
    _grpo_config,
    load_config,
    truncate_dataset_prompts,
    truncate_prompt_messages,
)


class TurnTrainerConfigTest(unittest.TestCase):
    @dataclass
    class FakeGRPOConfig:
        use_vllm: bool = False
        vllm_mode: str = "colocate"
        vllm_server_base_url: str | None = None
        vllm_server_host: str = "0.0.0.0"
        vllm_server_port: int = 8000
        vllm_server_timeout: float = 240.0
        vllm_group_port: int = 51216

    def test_loads_vllm_server_weight_sync_settings(self):
        raw = {
            "use_vllm": True,
            "vllm_mode": "server",
            "vllm_server_base_url": None,
            "vllm_server_host": "127.0.0.1",
            "vllm_server_port": 8123,
            "vllm_server_timeout": 900.0,
            "vllm_group_port": 51217,
            "num_generations": 4,
            "generation_batch_size": 8,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "turn-grpo.yaml"
            config_path.write_text("test config\n", encoding="utf-8")
            fake_yaml = types.SimpleNamespace(safe_load=lambda handle: raw)
            with patch.dict(sys.modules, {"yaml": fake_yaml}):
                cfg = load_config(str(config_path))

        self.assertTrue(cfg.use_vllm)
        self.assertEqual(cfg.vllm_mode, "server")
        self.assertEqual(cfg.vllm_server_host, "127.0.0.1")
        self.assertEqual(cfg.vllm_server_port, 8123)
        self.assertEqual(cfg.vllm_server_timeout, 900.0)
        self.assertEqual(cfg.vllm_group_port, 51217)
        self.assertEqual(cfg.num_generations, 4)
        self.assertEqual(cfg.generation_batch_size, 8)

    def test_forwards_vllm_server_settings_to_grpo(self):
        cfg = TurnGRPOTrainConfig(
            use_vllm=True,
            vllm_mode="server",
            vllm_server_base_url=None,
            vllm_server_host="127.0.0.1",
            vllm_server_port=8123,
            vllm_server_timeout=900.0,
            vllm_group_port=51217,
        )
        fake_trl = types.SimpleNamespace(GRPOConfig=self.FakeGRPOConfig)
        with (
            patch.dict(sys.modules, {"trl": fake_trl}),
            patch(
                "user_model.rl.trainer.version",
                return_value="test",
            ),
        ):
            args = _grpo_config(cfg)

        self.assertTrue(args.use_vllm)
        self.assertEqual(args.vllm_mode, "server")
        self.assertEqual(args.vllm_server_host, "127.0.0.1")
        self.assertEqual(args.vllm_server_port, 8123)
        self.assertEqual(args.vllm_server_timeout, 900.0)
        self.assertEqual(args.vllm_group_port, 51217)

    def test_rejects_invalid_vllm_server_port(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "turn-grpo.yaml"
            config_path.write_text("test config\n", encoding="utf-8")
            fake_yaml = types.SimpleNamespace(
                safe_load=lambda handle: {"vllm_server_port": 70000}
            )
            with patch.dict(sys.modules, {"yaml": fake_yaml}):
                with self.assertRaisesRegex(ValueError, "vllm_server_port"):
                    load_config(str(config_path))

    def test_gdpo_uses_trl_native_normalize_then_sum(self):
        @dataclass
        class FakeNativeGRPOConfig:
            multi_objective_aggregation: str = "sum_then_normalize"
            reward_weights: list[float] | None = None

        cfg = TurnGRPOTrainConfig(advantage_formula="gdpo")
        with (
            patch.dict(
                sys.modules,
                {"trl": types.SimpleNamespace(GRPOConfig=FakeNativeGRPOConfig)},
            ),
            patch("user_model.rl.trainer.version", return_value="test"),
        ):
            args = _grpo_config(cfg)

        self.assertEqual(args.multi_objective_aggregation, "normalize_then_sum")
        self.assertEqual(args.reward_weights, [1.0, 1.0, 1.0])

    def test_gdpo_rejects_trl_without_native_support(self):
        @dataclass
        class OldGRPOConfig:
            reward_weights: list[float] | None = None

        cfg = TurnGRPOTrainConfig(advantage_formula="gdpo")
        with patch.dict(
            sys.modules, {"trl": types.SimpleNamespace(GRPOConfig=OldGRPOConfig)}
        ):
            with self.assertRaisesRegex(RuntimeError, "TRL >= 1.7.0"):
                _grpo_config(cfg)


class _FakeSimilarity:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score_many(self, predicted, reference):
        self.calls.append((predicted, reference))
        repeats = (len(predicted) + len(self.scores) - 1) // len(self.scores)
        return (self.scores * repeats)[: len(predicted)]


class _FakeJudge:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score_many(self, rows):
        self.calls.append(rows)
        repeats = (len(rows) + len(self.scores) - 1) // len(self.scores)
        return (self.scores * repeats)[: len(rows)]


class DataPrepTurnsTest(unittest.TestCase):
    def test_preserves_simplified_html_as_ui(self):
        html = """<html><body>
          <form aria-label="Site search">
            <input type="search" placeholder="Find products"
              parser-semantic-id="search_box" parser-can-edit="true">
          </form>
          <a href="/products/tea" parser-semantic-id="tea" parser-clickable="true">
            <img alt="Green tea"><span>Organic tea</span>
          </a>
          <div hidden><button parser-semantic-id="secret">Hidden</button></div>
        </body></html>"""
        self.assertEqual(extract_ui(f"# context\n{html}"), html)

    def test_flattens_each_assistant_turn_without_leaking_its_label(self):
        first = json.dumps(
            {"rationale": "Open tea.", "action": "click", "target": "tea"}
        )
        second = json.dumps(
            {"rationale": "I am done.", "action": "terminate"}
        )
        messages = [
            {"role": "system", "content": "shop"},
            {"role": "user", "content": "<button parser-semantic-id='tea'>Tea</button>"},
            {"role": "assistant", "content": first},
            {"role": "user", "content": "<p>Tea details</p>"},
            {"role": "assistant", "content": second},
        ]
        rows = conversation_to_turns({"session_id": "s#w0", "messages": messages})
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["prompt"], messages[:2])
        self.assertNotIn(first, [m["content"] for m in rows[0]["prompt"]])
        self.assertEqual(rows[1]["prompt"], messages[:4])
        self.assertEqual(rows[0]["gt_target"], "tea")
        self.assertEqual(rows[1]["gt_target"], "")
        self.assertEqual(
            json.loads(rows[0]["ground_truth_completion"]), json.loads(first)
        )
        self.assertEqual(
            json.loads(rows[1]["ground_truth_completion"]), json.loads(second)
        )
        self.assertEqual(rows[0]["ui"], messages[1]["content"])
        self.assertEqual(rows[1]["ui"], messages[3]["content"])

    def test_already_flattened_hub_rows_are_not_flattened_twice(self):
        row = {
            "prompt": [{"role": "user", "content": "<p>Done</p>"}],
            "reference_rationale": "I am done.",
            "gt_action": "terminate",
            "gt_target": "",
            "ui": "<p>Done</p>",
            "session_id": "s#w0",
            "turn_idx": 0,
        }
        self.assertEqual(records_to_turns([row]), [row])

    def test_normalized_hub_data_is_sampled_per_store_and_written_locally(self):
        user_rows = [
            {
                "store_id": store_id,
                "session_id": f"session-{store_id}",
                "persona": "careful shopper",
                "intent": "inspect tea",
            }
            for store_id in ("store-a", "store-b")
        ]
        action_rows = []
        for user in user_rows:
            for turn_idx in range(2):
                action_rows.append(
                    {
                        "store_id": user["store_id"],
                        "session_id": user["session_id"],
                        "timestamp": f"2026-01-01T00:00:0{turn_idx}Z",
                        "action_json": json.dumps(
                            {
                                "rationale": "Inspect tea.",
                                "action": "click",
                                "target": f"tea-{turn_idx}",
                            }
                        ),
                        "simplified_dom": (
                            f"<button parser-semantic-id='tea-{turn_idx}'>Tea</button>"
                        ),
                    }
                )

        calls = []

        def fake_load_dataset(repo, config, split):
            calls.append((repo, config, split))
            return action_rows if config == "action" else user_rows

        fake_module = types.SimpleNamespace(load_dataset=fake_load_dataset)
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(sys.modules, {"datasets": fake_module}):
                output_path = prepare_turn_data(
                    input_repo="<huggingface_repo_id>/buyer-sim-50",
                    input_split="train+test",
                    output_dir=temp_dir,
                )
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            manifest = json.loads(
                (Path(temp_dir) / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(output_path, Path(temp_dir) / "train.jsonl")
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["store_id"] for row in rows}, {"store-a", "store-b"})
            self.assertEqual(
                {row["ui"] for row in rows},
                {
                    "<button parser-semantic-id='tea-0'>Tea</button>",
                    "<button parser-semantic-id='tea-1'>Tea</button>",
                },
            )
            self.assertTrue(all("ui_facts" not in row for row in rows))
            self.assertEqual(manifest["session_count"], 2)
            self.assertEqual(manifest["turn_count"], 4)
            self.assertEqual(manifest["input_repo"], "<huggingface_repo_id>/buyer-sim-50")
            self.assertEqual(manifest["input_split"], "train+test")
            self.assertFalse((Path(temp_dir) / "mixed").exists())
        self.assertEqual(
            calls,
            [
                ("<huggingface_repo_id>/buyer-sim-50", "action", "train+test"),
                ("<huggingface_repo_id>/buyer-sim-50", "user", "train+test"),
            ],
        )

    def test_legacy_synthetic_keywords_remain_supported(self):
        user_rows = [
            {
                "store_id": "store-a",
                "session_id": "session-a",
                "persona": "shopper",
                "intent": "browse",
            }
        ]
        action_rows = [
            {
                "store_id": "store-a",
                "session_id": "session-a",
                "timestamp": str(index),
                "action_json": json.dumps(
                    {"rationale": "Browse.", "action": "click", "target": str(index)}
                ),
                "simplified_dom": "<button>Browse</button>",
            }
            for index in range(2)
        ]

        def fake_load_dataset(repo, config, split):
            self.assertEqual(repo, "legacy/repo")
            self.assertEqual(split, "train")
            return action_rows if config == "action" else user_rows

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                sys.modules,
                {"datasets": types.SimpleNamespace(load_dataset=fake_load_dataset)},
            ):
                output = prepare_turn_data(
                    synthetic_repo="legacy/repo",
                    synthetic_split="train",
                    output_dir=temp_dir,
                )
            self.assertTrue(output.is_file())


class PromptTruncationTest(unittest.TestCase):
    class FakeTokenizer:
        def apply_chat_template(
            self, messages, tokenize, add_generation_prompt, **kwargs
        ):
            self.assertions = (tokenize, add_generation_prompt)
            return " ".join(message["content"] for message in messages) + " <generate>"

        def __call__(self, rendered, add_special_tokens):
            self.add_special_tokens = add_special_tokens
            input_ids = list(range(len(rendered.split())))
            # Match transformers.BatchEncoding: a Mapping/UserDict whose len()
            # is the number of fields, not the input token count.
            return UserDict(
                {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
            )

    def test_drops_oldest_complete_exchange_and_keeps_current_state(self):
        messages = [
            {"role": "system", "content": "persona and intent"},
            {"role": "user", "content": "old page with products"},
            {"role": "assistant", "content": "old click action"},
            {"role": "user", "content": "current page search bar"},
        ]
        tokenizer = self.FakeTokenizer()
        prompt, before, after, dropped = truncate_prompt_messages(
            tokenizer, messages, max_tokens=8
        )
        self.assertGreater(before, 8)
        self.assertLessEqual(after, 8)
        self.assertEqual(dropped, 2)
        self.assertEqual([message["role"] for message in prompt], ["system", "user"])
        self.assertEqual(prompt[-1], messages[-1])
        self.assertEqual(tokenizer.assertions, (False, True))
        self.assertFalse(tokenizer.add_special_tokens)

    def test_oversized_current_observation_fails_instead_of_silent_crop(self):
        messages = [
            {"role": "system", "content": "persona intent"},
            {"role": "user", "content": "one two three four five six seven"},
        ]
        with self.assertRaisesRegex(ValueError, "current observation requires"):
            truncate_prompt_messages(self.FakeTokenizer(), messages, max_tokens=5)

    def test_persistent_prompt_cache_skips_retokenization(self):
        class CountingTokenizer(self.FakeTokenizer):
            name_or_path = "fake-checkpoint"
            chat_template = "fake-template"
            special_tokens_map = {"eos_token": "<eos>"}

            def __init__(self):
                self.calls = 0

            def __call__(self, rendered, add_special_tokens):
                self.calls += 1
                return super().__call__(rendered, add_special_tokens)

        class FakeDataset:
            cached_rows = {}

            def __init__(self, rows, fingerprint="same-source"):
                self.rows = rows
                self._fingerprint = fingerprint
                self.column_names = list(rows[0]) if rows else []

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, key):
                if isinstance(key, str):
                    return [row[key] for row in self.rows]
                return self.rows[key]

            def map(self, function, **kwargs):
                cache_file = kwargs.get("cache_file_name")
                if (
                    kwargs.get("load_from_cache_file")
                    and cache_file in self.cached_rows
                ):
                    rows = [dict(row) for row in self.cached_rows[cache_file]]
                else:
                    rows = []
                    for index, row in enumerate(self.rows):
                        updated = dict(row)
                        updated.update(function(dict(row), index))
                        rows.append(updated)
                    if cache_file:
                        self.cached_rows[cache_file] = [dict(row) for row in rows]
                        Path(cache_file).touch()
                return FakeDataset(rows, self._fingerprint)

            def remove_columns(self, columns):
                return FakeDataset(
                    [
                        {key: value for key, value in row.items() if key not in columns}
                        for row in self.rows
                    ],
                    self._fingerprint,
                )

        class FakePartialState:
            is_main_process = True

            def main_process_first(self):
                from contextlib import nullcontext

                return nullcontext()

        messages = [
            {"role": "system", "content": "persona and intent"},
            {"role": "user", "content": "old page with products"},
            {"role": "assistant", "content": "old click action"},
            {"role": "user", "content": "current page search bar"},
        ]
        dataset = FakeDataset([{"prompt": messages, "session_id": "s1", "turn_idx": 1}])
        fake_accelerate = types.SimpleNamespace(PartialState=FakePartialState)

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            sys.modules, {"accelerate": fake_accelerate}
        ):
            first_tokenizer = CountingTokenizer()
            first = truncate_dataset_prompts(
                dataset, first_tokenizer, 8, "train", cache_dir=temp_dir
            )
            self.assertGreater(first_tokenizer.calls, 0)

            second_tokenizer = CountingTokenizer()
            second = truncate_dataset_prompts(
                dataset, second_tokenizer, 8, "train", cache_dir=temp_dir
            )
            self.assertEqual(second_tokenizer.calls, 0)
            self.assertEqual(first.rows, second.rows)
            self.assertFalse(
                set(second.column_names).intersection(
                    {
                        "__prompt_cap_before_tokens",
                        "__prompt_cap_after_tokens",
                        "__prompt_cap_dropped_messages",
                    }
                )
            )


class TurnFormulaRewardTest(unittest.TestCase):
    @staticmethod
    def _kwargs(action="click", target="product-card-123"):
        return {
            "gt_action": [action],
            "gt_target": [target],
            "ui": [
                '<button parser-semantic-id="product-card">Product</button>'
            ],
        }

    def test_target_score_is_ground_truth_substring_percentage(self):
        self.assertEqual(
            target_substring_score("product-card-123", "product-card-123"), 1.0
        )
        self.assertAlmostEqual(
            target_substring_score(
                "nav_dir.search_button", "nav_dir.search_input"
            ),
            len("nav_dir.search_")
            / ((len("nav_dir.search_button") + len("nav_dir.search_input")) / 2),
        )
        self.assertAlmostEqual(
            target_substring_score(
                "abcd.asd.qwe", "asd.qwe"
            ),
            len("asd.qwe") / ((len("abcd.asd.qwe") + len("asd.qwe")) / 2),
        )
        self.assertEqual(target_substring_score("xyz", "abc"), 0.0)
        self.assertEqual(target_substring_score("", ""), 1.0)

    def test_target_reward_is_one_weighted_scalar(self):
        ui = (
            '<button parser-semantic-id="product-card">Product</button>'
            '<button data-semantic-id="exact-card">Exact</button>'
        )
        lcs = target_substring_score("product-card", "product-card-123")
        self.assertAlmostEqual(
            target_reward_score("product-card", "product-card-123", ui),
            0.60 * lcs + 0.10,
        )
        self.assertAlmostEqual(
            target_reward_score("exact-card", "exact-card", ui),
            0.60 + 0.10 + 0.30,
        )

    def test_constraint_aware_formula(self):
        reward = ConstraintAwareReward(
            TurnFormulaRewardConfig(
                format_weight=2.0,
                action_weight=0.6,
                target_weight=0.4,
            )
        )
        valid = json.dumps(
            {
                "rationale": "Inspect the product.",
                "action": "click",
                "target": "product-card",
            }
        )
        wrong_action = json.dumps(
            {
                "rationale": "Inspect the product.",
                "action": "hover",
                "target": "product-card",
            }
        )
        components = reward.score_components(
            [valid, wrong_action, "not json"],
            gt_action=["click"] * 3,
            gt_target=["product-card-123"] * 3,
            ui=[
                '<button parser-semantic-id="product-card">Product</button>'
            ] * 3,
        )
        lcs = len("product-card") / (
            (len("product-card") + len("product-card-123")) / 2
        )
        expected_target = 0.60 * lcs + 0.10
        self.assertAlmostEqual(components[0]["target"], expected_target)
        self.assertAlmostEqual(
            components[0]["total"], 0.6 + 0.4 * expected_target
        )
        self.assertAlmostEqual(components[1]["total"], 0.4 * expected_target)
        self.assertEqual(components[2]["total"], -2.0)

    def test_constraint_weights_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "must equal 1"):
            TurnFormulaRewardConfig(action_weight=0.7, target_weight=0.4).validate()

    def test_gdpo_builds_three_separate_reward_functions(self):
        rewards = build_formula_rewards("gdpo", TurnFormulaRewardConfig())
        self.assertEqual(
            [reward.__name__ for reward in rewards],
            ["R_format", "R_action", "R_target"],
        )
        completion = json.dumps(
            {
                "rationale": "Inspect the product.",
                "action": "click",
                "target": "product-card",
            }
        )
        scores = [
            reward([completion], **self._kwargs())[0]
            for reward in rewards
        ]
        self.assertEqual(scores[:2], [1.0, 1.0])
        self.assertAlmostEqual(
            scores[2],
            0.60
            * len("product-card")
            / ((len("product-card") + len("product-card-123")) / 2)
            + 0.10,
        )


class LegacyTurnRewardTest(unittest.TestCase):
    def _reward(self, ui=(1.0,), similarity=(1.0,)):
        judge = _FakeJudge(list(ui))
        scorer = _FakeSimilarity(list(similarity))
        reward = TurnCompositeReward(
            TurnRewardConfig(), ui_judge=judge, reasoning_scorer=scorer
        )
        return reward, judge, scorer

    @staticmethod
    def _kwargs(action="click", target="tea"):
        return {
            "reference_rationale": ["I want to inspect the tea."],
            "gt_action": [action],
            "gt_target": [target],
            "ui": ["<button parser-semantic-id='tea'>Tea</button>"],
        }

    def test_exact_target_is_the_final_anchor(self):
        reward, judge, _ = self._reward()
        correct = json.dumps(
            {"rationale": "Inspect tea.", "action": "click", "target": "tea"}
        )
        wrong = json.dumps(
            {"rationale": "Inspect tea.", "action": "click", "target": "xyz"}
        )
        components = reward.score_components([correct, wrong], **{
            key: value * 2 for key, value in self._kwargs().items()
        })
        self.assertEqual(components[0]["target"], 1.0)
        self.assertEqual(components[0]["total"], 1.0)
        self.assertEqual(components[1]["target"], 0.0)
        self.assertEqual(components[1]["total"], 0.5)
        self.assertEqual(
            judge.calls[0][0][0],
            "<button parser-semantic-id='tea'>Tea</button>",
        )

    def test_format_and_action_are_hard_gates_and_skip_judge(self):
        reward, judge, scorer = self._reward()
        malformed = "not json"
        wrong_action = json.dumps(
            {"rationale": "Inspect tea.", "action": "hover", "target": "tea"}
        )
        totals = reward([malformed, wrong_action], **{
            key: value * 2 for key, value in self._kwargs().items()
        })
        self.assertEqual(totals, [0.0, 0.0])
        self.assertEqual(judge.calls, [])
        self.assertEqual(scorer.calls, [])

    def test_action_type_match_is_case_sensitive_exact_match(self):
        reward, judge, scorer = self._reward()
        noncanonical = json.dumps(
            {"rationale": "Inspect tea.", "action": "Click", "target": "tea"}
        )
        component = reward.score_components([noncanonical], **self._kwargs())[0]
        self.assertEqual(component["format"], 0.0)
        self.assertEqual(component["action"], 0.0)
        self.assertEqual(component["total"], 0.0)
        self.assertEqual(judge.calls, [])
        self.assertEqual(scorer.calls, [])

    def test_targetless_action_omits_and_renormalizes_target_weight(self):
        reward, _, _ = self._reward(ui=(0.6,), similarity=(0.4,))
        completion = json.dumps(
            {"rationale": "I am finished browsing.", "action": "terminate"}
        )
        component = reward.score_components(
            [completion], **self._kwargs(action="terminate", target="")
        )[0]
        # reasoning=.7*.6+.3*.4=.54; (.05+.45*.54)/(.05+.45)=.586
        self.assertAlmostEqual(component["reasoning"], 0.54)
        self.assertAlmostEqual(component["total"], 0.586)

    def test_unrepeated_dataset_columns_align_to_generation_groups(self):
        reward, _, _ = self._reward(ui=(1.0, 1.0), similarity=(1.0, 1.0))
        completions = [
            json.dumps({"rationale": "A", "action": "click", "target": "a"}),
            json.dumps({"rationale": "A2", "action": "click", "target": "a"}),
            json.dumps({"rationale": "B", "action": "click", "target": "b"}),
            json.dumps({"rationale": "B2", "action": "click", "target": "b"}),
        ]
        totals = reward(
            completions,
            reference_rationale=["A", "B"],
            gt_action=["click", "click"],
            gt_target=["a", "b"],
            ui=[
                "<button parser-semantic-id='a'>A</button>",
                "<button parser-semantic-id='b'>B</button>",
            ],
        )
        self.assertEqual(totals, [1.0, 1.0, 1.0, 1.0])

    def test_logs_reward_components_and_ground_truth_to_completion_table(self):
        reward, _, _ = self._reward(ui=(0.8, 0.6), similarity=(0.4, 0.2))
        completions = [
            json.dumps(
                {"rationale": "Inspect tea.", "action": "click", "target": "tea"}
            ),
            json.dumps(
                {
                    "rationale": "Inspect another item.",
                    "action": "click",
                    "target": "xyz",
                }
            ),
        ]
        logged = {}
        totals = reward(
            completions,
            **{
                key: value * 2 for key, value in self._kwargs().items()
            },
            session_id=["s1", "s1"],
            turn_idx=[3, 3],
            log_extra=lambda column, values: logged.update({column: values}),
        )

        self.assertEqual(logged["format_valid"], [True, True])
        self.assertEqual(logged["action_match"], [True, True])
        self.assertEqual(logged["target_match"], [True, False])
        self.assertEqual(logged["target_applicable"], [True, True])
        self.assertEqual(logged["ui_judge_reward"], [0.8, 0.6])
        self.assertAlmostEqual(logged["reasoning_reward"][0], 0.68)
        self.assertAlmostEqual(logged["reasoning_reward"][1], 0.48)
        self.assertEqual(logged["ground_truth_action"], ["click", "click"])
        self.assertEqual(logged["ground_truth_target"], ["tea", "tea"])
        self.assertEqual(json.loads(logged["ground_truth"][0])["target"], "tea")
        self.assertEqual(logged["zero_reward"], [False, False])
        self.assertEqual(totals, [0.856, 0.266])

    def test_ui_judge_reports_failure_separately_from_zero_score(self):
        cfg = TurnRewardConfig(
            ui_judge_failure_mode="zero",
            ui_judge_max_workers=1,
        )
        failed_judge = GeminiUIGroundingJudge(cfg, call_fn=lambda messages: "")
        scores, failures = failed_judge.score_many_with_status(
            [("<button>Tea</button>", "Inspect tea.", "click", "tea")]
        )
        self.assertEqual(scores, [0.0])
        self.assertEqual(failures, [True])

        zero_judge = GeminiUIGroundingJudge(
            cfg, call_fn=lambda messages: '{"score": 0.0}'
        )
        scores, failures = zero_judge.score_many_with_status(
            [("<button>Tea</button>", "Inspect tea.", "click", "tea")]
        )
        self.assertEqual(scores, [0.0])
        self.assertEqual(failures, [False])


class TurnGroupDiagnosticsTest(unittest.TestCase):
    def test_logs_each_response_in_two_generation_groups(self):
        class BaseTrainer:
            def _generate_and_score_completions(self, inputs):
                return {"advantages": "local tensor"}

        class FakeTrainer(_TurnGroupDiagnosticsMixin, BaseTrainer):
            pass

        trainer = FakeTrainer()
        trainer.accelerator = types.SimpleNamespace(
            is_main_process=True, num_processes=1
        )
        trainer.args = types.SimpleNamespace(report_to=[])
        trainer.state = types.SimpleNamespace(global_step=0)
        trainer.model = types.SimpleNamespace(training=True)
        trainer.log_group_diagnostics = True
        trainer.num_generations = 2
        trainer.num_generations_eval = 2
        trainer.reward_func_names = ["turn_total"]
        trainer._metrics = {
            "train": defaultdict(list),
            "eval": defaultdict(list),
        }
        trainer._logs = {
            "prompt": ["prompt-a", "prompt-a", "prompt-b", "prompt-b"],
            "completion": ["a1", "a2", "b1", "b2"],
            "rewards": {"turn_total": [0.1, 0.9, 0.4, 0.6]},
            "advantages": [-0.7, 0.7, -0.7, 0.7],
            "extra": {
                "session_id": ["s1", "s1", "s2", "s2"],
                "turn_idx": [3, 3, 4, 4],
                "format_valid": [True, True, False, True],
                "action_match": [True, True, False, True],
                "target_applicable": [True, True, False, True],
                "target_match": [False, True, False, True],
                "ui_judge_reward": [0.2, 0.8, 0.0, 0.6],
                "reasoning_reward": [0.3, 0.7, 0.0, 0.5],
                "ui_judge_attempted": [True, True, False, True],
                "ui_judge_failed": [False, True, False, False],
                "zero_reward": [False, False, True, False],
            },
        }

        with self.assertLogs(
            "user_model.rl.trainer", level=logging.INFO
        ) as captured:
            result = trainer._generate_and_score_completions([{}, {}, {}, {}])

        output = "\n".join(captured.output)
        self.assertEqual(result, {"advantages": "local tensor"})
        self.assertIn("4 responses = 2 groups x 2 generations", output)
        self.assertIn("GRPO group 1/2 (session_id='s1', turn_idx=3)", output)
        self.assertIn(
            'response 2/2 reward=0.9000000000 advantage=+0.7000000000 completion="a2"',
            output,
        )
        self.assertEqual(trainer._metrics["train"]["rollout/rewards"], [0.5])
        self.assertEqual(trainer._metrics["train"]["rollout/advantages"], [0.7])
        self.assertAlmostEqual(
            trainer._metrics["train"]["rollout/advantages_signed_mean"][0],
            0.0,
        )
        self.assertEqual(
            trainer._metrics["train"]["rollout/format_valid_rate"], [0.75]
        )
        self.assertEqual(
            trainer._metrics["train"]["rollout/action_match_rate"], [0.75]
        )
        self.assertEqual(
            trainer._metrics["train"]["rollout/target_match_rate"], [2 / 3]
        )
        self.assertEqual(
            trainer._metrics["train"]["rollout/ui_judge_reward"], [1.6 / 3]
        )
        self.assertEqual(
            trainer._metrics["train"]["rollout/reasoning_reward"], [0.5]
        )
        self.assertEqual(
            trainer._metrics["train"]["rollout/judge_failure_rate"], [1 / 3]
        )
        self.assertEqual(
            trainer._metrics["train"]["rollout/zero_reward_fraction"], [0.25]
        )


if __name__ == "__main__":
    unittest.main()
