"""Phase 5 provider-neutral prompt and contract tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lm_experiments.lm_arms.client import (
    build_request,
    call_lm_arm,
    validate_prediction,
)
from lm_experiments.lm_arms.tools import FIND_COMPARABLES_SCHEMA, ToolBox
from lm_experiments.lm_arms.prompts import render


VALUES = {
    "anon_id": "player_deadbeef",
    "player_name": "MASK_TEST_PLAYER",
    "season_t": 2024,
    "season_t1": 2025,
    "vec_t_power": 60.0,
    "vec_t_contact": 55.0,
    "vec_t_discipline": 50.0,
    "vec_t_defense": 45.0,
    "vec_t_speed": 40.0,
    "delta_vec_power": 1.0,
    "delta_vec_contact": 2.0,
    "delta_vec_discipline": 3.0,
    "delta_vec_defense": 4.0,
    "delta_vec_speed": 5.0,
}


class PromptBoundaryTests(unittest.TestCase):
    def test_blind_arms_mask_name_and_seasons(self) -> None:
        for arm in ("b", "d"):
            prompt = render(arm, **VALUES)
            self.assertNotIn(VALUES["player_name"], prompt)
            self.assertNotIn("2024", prompt)
            self.assertNotIn("2025", prompt)
            self.assertNotIn("WAR:", prompt)

    def test_named_arm_exposes_name_and_seasons(self) -> None:
        prompt = render("c", **VALUES)
        self.assertIn(VALUES["player_name"], prompt)
        self.assertIn("2024", prompt)
        self.assertIn("2025", prompt)

    def test_arm_tool_boundaries(self) -> None:
        b = build_request("local_record", "b", render("b", **VALUES))
        d = build_request("local_record", "d", render("d", **VALUES))
        self.assertEqual([tool["name"] for tool in b["tools"]], ["submit_prediction"])
        self.assertEqual(len(d["tools"]), 6)


class PredictionContractTests(unittest.TestCase):
    def test_prediction_normalization(self) -> None:
        result = validate_prediction(
            {"delta_war": 1, "delta_wrc": 2, "delta_ops": 0.03}, "record_1"
        )
        self.assertEqual(result["record_id"], "record_1")
        self.assertIsNone(result["confidence"])

    def test_invalid_prediction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_prediction(
                {"delta_war": 1, "delta_wrc": 2, "delta_ops": 0.03, "confidence": 2}
            )

    def test_api_call_is_deliberately_unimplemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            call_lm_arm()


class ComparableInputTests(unittest.TestCase):
    def test_find_schema_requires_scouting_scale(self) -> None:
        properties = FIND_COMPARABLES_SCHEMA["input_schema"]["properties"]["vec5"]["properties"]
        for value_schema in properties.values():
            self.assertEqual(value_schema["minimum"], 20)
            self.assertEqual(value_schema["maximum"], 80)

    def test_delta_vector_is_rejected_as_comparable_query(self) -> None:
        toolbox = ToolBox.__new__(ToolBox)
        toolbox.tool_order = ["Power", "Contact", "Discipline", "Defense", "Speed"]
        with self.assertRaisesRegex(ValueError, "delta vector"):
            toolbox.find_comparables(
                {"Power": -8.4, "Contact": -1.2, "Discipline": -4.8, "Defense": -2.4, "Speed": 9.5},
                k=10,
            )


if __name__ == "__main__":
    unittest.main()
