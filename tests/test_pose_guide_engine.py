from __future__ import annotations

import math
import unittest

from src.models.schemas import PoseGuideRecommendation, PoseTemplateId
from src.pose_guide_engine import PoseGuideEngine
from tests.test_marker_renderer import synthetic_coco17_pose


TEMPLATES: tuple[PoseTemplateId, ...] = (
    "seated_relaxed",
    "seated_open_shoulders",
    "seated_hand_on_thigh",
    "seated_side_lean",
    "standing_relaxed",
    "standing_contrapposto",
    "hand_on_hip",
    "arms_open_soft",
    "walking_stride_soft",
    "leaning_side",
    "portrait_chin_angle",
    "one_hand_near_face",
)


def guide(template_id: PoseTemplateId) -> PoseGuideRecommendation:
    return PoseGuideRecommendation(
        pose_template_id=template_id,
        name=f"{template_id} guide",
        category="sitting" if template_id.startswith("seated") else "standing",
        description="Test guide",
        reasoning="Test reasoning",
        difficulty="easy",
        camera_angle_suggestion="eye-level",
        guide_params={},
        correction_focus=["shoulders"],
    )


class PoseGuideEngineTests(unittest.TestCase):
    def test_every_template_generates_valid_target_pose(self) -> None:
        pose = synthetic_coco17_pose()
        engine = PoseGuideEngine()
        for template_id in TEMPLATES:
            with self.subTest(template_id=template_id):
                target = engine.generate(pose, guide(template_id))
                self.assertGreaterEqual(len(target.target_keypoints), 16)

    def test_seated_pose_remaps_extreme_standing_template(self) -> None:
        pose = seated_coco17_pose()
        engine = PoseGuideEngine()
        target = engine.generate(pose, guide("walking_stride_soft"))
        self.assertEqual(target.pose_template_id, "seated_open_shoulders")
        self.assertLess(target.target_keypoints["left_ankle"].y, 0.99)

    def test_generated_pose_changes_current_wrist_without_teleporting(self) -> None:
        pose = synthetic_coco17_pose()
        engine = PoseGuideEngine()
        target = engine.generate(pose, guide("seated_open_shoulders"))
        current = pose.keypoints["left_wrist"]
        generated = target.target_keypoints["left_wrist"]
        distance = math.hypot(generated.x - current.x, generated.y - current.y)
        self.assertGreater(distance, 0.02)
        self.assertLess(distance, 0.45)


if __name__ == "__main__":
    unittest.main()


def seated_coco17_pose():
    pose = synthetic_coco17_pose()
    updates = {
        "left_hip": 0.78,
        "right_hip": 0.78,
        "left_knee": 0.92,
        "right_knee": 0.92,
        "left_ankle": 0.97,
        "right_ankle": 0.97,
    }
    keypoints = dict(pose.keypoints)
    for name, y in updates.items():
        landmark = keypoints[name]
        keypoints[name] = landmark.model_copy(
            update={
                "y": y,
                "pixel_y": int(round(y * (pose.image_height - 1))),
            }
        )
    return pose.model_copy(update={"keypoints": keypoints})
