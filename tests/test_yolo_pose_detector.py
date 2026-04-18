from __future__ import annotations

import unittest

import numpy as np

from src.yolo_pose_detector import COCO17_LANDMARK_NAMES, yolo_result_to_pose_data


class FakeKeypoints:
    def __init__(self, data):
        self.data = data


class FakeResult:
    def __init__(self, data):
        self.keypoints = FakeKeypoints(data)


class YoloPoseDetectorTests(unittest.TestCase):
    def test_yolo_coco17_mapping_creates_pose_data(self) -> None:
        data = np.zeros((1, 17, 3), dtype=float)
        for index in range(17):
            data[0, index, 0] = 10 + index
            data[0, index, 1] = 20 + index
            data[0, index, 2] = 0.8

        pose = yolo_result_to_pose_data(FakeResult(data), image_width=100, image_height=200)

        self.assertEqual(pose.keypoint_profile, "coco17")
        self.assertEqual(set(pose.keypoints.keys()), set(COCO17_LANDMARK_NAMES))
        self.assertAlmostEqual(pose.keypoints["left_shoulder"].x, 15 / 100)
        self.assertAlmostEqual(pose.keypoints["left_shoulder"].y, 25 / 200)
        self.assertAlmostEqual(pose.confidence, 0.8)


if __name__ == "__main__":
    unittest.main()

