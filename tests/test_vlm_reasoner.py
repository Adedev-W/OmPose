from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from src.models.prompts import build_repair_prompt
from src.vlm_reasoner import extract_json_text, parse_pose_guide_result, parse_reasoning_result
from tests.test_schemas import VALID_GUIDE_PAYLOAD, VALID_PAYLOAD


class VLMParserTests(unittest.TestCase):
    def test_plain_json_parses(self) -> None:
        parsed = parse_reasoning_result(json.dumps(VALID_PAYLOAD))
        self.assertEqual(parsed.recommendations[0].name, "Open Scenic Pose")

    def test_fenced_json_is_extracted(self) -> None:
        raw = "```json\n" + json.dumps(VALID_PAYLOAD) + "\n```"
        self.assertEqual(extract_json_text(raw), json.dumps(VALID_PAYLOAD))
        parsed = parse_reasoning_result(raw)
        self.assertEqual(parsed.scene.scene_type, "outdoor")

    def test_invalid_json_raises_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            parse_reasoning_result('{"scene": {}, "recommendations": []}')

    def test_repair_prompt_includes_validation_error(self) -> None:
        prompt = build_repair_prompt('{"bad": true}', "missing pose_template_id")
        self.assertIn("missing pose_template_id", prompt)
        self.assertIn("pose_template_id", prompt)

    def test_pose_guide_json_parses(self) -> None:
        parsed = parse_pose_guide_result(json.dumps(VALID_GUIDE_PAYLOAD))
        self.assertEqual(parsed.recommendation.pose_template_id, "seated_open_shoulders")


if __name__ == "__main__":
    unittest.main()
