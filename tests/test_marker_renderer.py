from __future__ import annotations

import unittest

import numpy as np

from src.marker_renderer import BODY_JOINTS, MarkerRenderer
from src.models.schemas import PoseData, PoseLandmark, PoseRecommendation
from src.pose_detector import LANDMARK_NAMES
from src.yolo_pose_detector import COCO17_LANDMARK_NAMES
from tests.test_schemas import TARGET_KEYPOINTS


def synthetic_pose(width: int = 320, height: int = 480) -> PoseData:
    coordinates = {
        "nose": (160, 80),
        "left_shoulder": (125, 145),
        "right_shoulder": (195, 145),
        "left_elbow": (105, 210),
        "right_elbow": (215, 210),
        "left_wrist": (95, 270),
        "right_wrist": (225, 270),
        "left_hip": (135, 285),
        "right_hip": (185, 285),
        "left_knee": (130, 360),
        "right_knee": (190, 360),
        "left_ankle": (125, 440),
        "right_ankle": (195, 440),
        "left_heel": (118, 452),
        "right_heel": (202, 452),
        "left_foot_index": (105, 454),
        "right_foot_index": (215, 454),
    }
    keypoints = {}
    for index, name in enumerate(LANDMARK_NAMES):
        x, y = coordinates.get(name, (160, 240))
        keypoints[name] = PoseLandmark(
            index=index,
            name=name,
            x=x / width,
            y=y / height,
            z=0.0,
            pixel_x=x,
            pixel_y=y,
            visibility=0.99,
            presence=0.99,
        )
    return PoseData(image_width=width, image_height=height, keypoints=keypoints, confidence=0.99)


def synthetic_coco17_pose(width: int = 320, height: int = 480) -> PoseData:
    full = synthetic_pose(width=width, height=height)
    keypoints = {
        name: PoseLandmark(
            index=index,
            name=name,
            x=full.keypoints[name].x,
            y=full.keypoints[name].y,
            z=None,
            pixel_x=full.keypoints[name].pixel_x,
            pixel_y=full.keypoints[name].pixel_y,
            visibility=0.98,
            presence=0.98,
        )
        for index, name in enumerate(COCO17_LANDMARK_NAMES)
    }
    return PoseData(
        image_width=width,
        image_height=height,
        keypoint_profile="coco17",
        keypoints=keypoints,
        confidence=0.98,
    )


def recommendation(template: str) -> PoseRecommendation:
    return PoseRecommendation(
        name=f"{template} pose",
        category="standing" if template != "seated_lean" else "sitting",
        description="Test pose",
        reasoning="Test reasoning",
        keypoint_adjustments={},
        difficulty="easy",
        camera_angle_suggestion="eye-level",
        marker_template=template,
        marker_side="right",
        marker_intensity=1.0,
        target_keypoints=TARGET_KEYPOINTS,
        target_pose_quality_notes="Test target keypoints.",
    )


class MarkerRendererTests(unittest.TestCase):
    def test_target_keypoints_generate_body_points(self) -> None:
        pose = synthetic_pose()
        renderer = MarkerRenderer()
        for template in (
            "contrapposto",
            "hand_on_hip",
            "arms_open",
            "walking_stride",
            "leaning_side",
            "seated_lean",
        ):
            with self.subTest(template=template):
                points = renderer.target_points(pose.image_width, pose.image_height, recommendation(template))
                self.assertTrue(set(BODY_JOINTS).issubset(points.keys()))

    def test_overlay_is_non_empty(self) -> None:
        pose = synthetic_pose()
        renderer = MarkerRenderer()
        image = np.zeros((pose.image_height, pose.image_width, 3), dtype=np.uint8)
        overlay = renderer.render(image, pose, recommendation("arms_open"))
        self.assertEqual(overlay.shape, image.shape)
        self.assertGreater(int(overlay.sum()), 0)

    def test_target_points_do_not_copy_current_pose(self) -> None:
        pose = synthetic_pose()
        renderer = MarkerRenderer()
        current_points = renderer.current_pose_points(pose)
        target_points = renderer.target_points(pose.image_width, pose.image_height, recommendation("arms_open"))
        self.assertNotEqual(current_points["right_wrist"], target_points["right_wrist"])

    def test_contact_sheet_is_non_empty(self) -> None:
        pose = synthetic_pose()
        renderer = MarkerRenderer()
        image = np.zeros((pose.image_height, pose.image_width, 3), dtype=np.uint8)
        sheet = renderer.render_contact_sheet(
            image,
            pose,
            [recommendation("arms_open"), recommendation("hand_on_hip"), recommendation("walking_stride")],
        )
        self.assertGreater(sheet.shape[1], image.shape[1])
        self.assertGreater(int(sheet.sum()), 0)

    def test_stream_overlay_handles_coco17_current_pose(self) -> None:
        pose = synthetic_coco17_pose()
        renderer = MarkerRenderer()
        image = np.zeros((pose.image_height, pose.image_width, 3), dtype=np.uint8)
        overlay = renderer.render_stream_overlay(image, pose, recommendation("arms_open"))
        self.assertEqual(overlay.shape, image.shape)
        self.assertGreater(int(overlay.sum()), 0)


if __name__ == "__main__":
    unittest.main()
