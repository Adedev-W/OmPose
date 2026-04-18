from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import main
from src.errors import ConfigError
from src.pipeline import OmPosePipeline as RealOmPosePipeline
from tests.test_pipeline import FakePoseDetector, FakeVLMReasoner


class CLITests(unittest.TestCase):
    def test_missing_file_returns_exit_2(self) -> None:
        code = main.main(["/tmp/not-real-ompose.jpg"])
        self.assertEqual(code, 2)

    def test_missing_api_key_returns_exit_1(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "input.jpg"
            cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))
            with patch.dict(os.environ, {}, clear=True):
                with patch.object(main, "load_dotenv", lambda: None):
                    code = main.main([str(image_path)])
            self.assertEqual(code, ConfigError.exit_code)

    def test_happy_path_with_mocked_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "input.jpg"
            cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))

            def build_fake_pipeline(*args, **kwargs):
                return RealOmPosePipeline(
                    pose_detector=FakePoseDetector(),
                    vlm_reasoner=FakeVLMReasoner(),
                )

            with patch.object(main, "OmPosePipeline", side_effect=build_fake_pipeline):
                code = main.main([str(image_path), "--out", str(root / "out")])
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
