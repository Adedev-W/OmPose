from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.models.schemas import VLMReasoningResult
from src.pipeline import OmPosePipeline
from tests.test_marker_renderer import synthetic_pose
from tests.test_schemas import VALID_PAYLOAD


class FakePoseDetector:
    def detect(self, image_path: Path, pose_index: int = 0):
        return synthetic_pose(width=64, height=64)


class FakeVLMReasoner:
    def recommend(self, image_path: Path, current_pose, recommendation_count: int = 3):
        payload = copy.deepcopy(VALID_PAYLOAD)
        base = payload["recommendations"][0]
        payload["recommendations"] = [
            {**copy.deepcopy(base), "name": "Open Scenic Pose", "marker_template": "arms_open"},
            {**copy.deepcopy(base), "name": "Hand Hip Pose", "marker_template": "hand_on_hip"},
            {**copy.deepcopy(base), "name": "Walking Pose", "marker_template": "walking_stride"},
        ]
        return VLMReasoningResult.model_validate(payload), {"total_tokens": 1}


class PipelineTests(unittest.TestCase):
    def test_pipeline_writes_overlay_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "input.jpg"
            out_dir = root / "out"
            cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))

            pipeline = OmPosePipeline(
                pose_detector=FakePoseDetector(),
                vlm_reasoner=FakeVLMReasoner(),
            )
            result = pipeline.run(image_path=image_path, out_dir=out_dir)

            self.assertTrue(Path(result.overlay_image).exists())
            self.assertTrue(Path(result.contact_sheet_image).exists())
            self.assertTrue(Path(result.result_json).exists())
            self.assertEqual(result.recommendations[0].marker_template, "arms_open")


if __name__ == "__main__":
    unittest.main()
