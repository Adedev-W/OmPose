from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.models.schemas import VLMPoseGuideResult, VLMReasoningResult


TARGET_KEYPOINTS = {
    "left_shoulder": {"x": 0.38, "y": 0.22, "visibility": 1.0},
    "right_shoulder": {"x": 0.62, "y": 0.22, "visibility": 1.0},
    "left_elbow": {"x": 0.30, "y": 0.36},
    "right_elbow": {"x": 0.74, "y": 0.30},
    "left_wrist": {"x": 0.25, "y": 0.48},
    "right_wrist": {"x": 0.80, "y": 0.18},
    "left_hip": {"x": 0.43, "y": 0.54},
    "right_hip": {"x": 0.57, "y": 0.54},
    "left_knee": {"x": 0.40, "y": 0.72},
    "right_knee": {"x": 0.62, "y": 0.72},
    "left_ankle": {"x": 0.36, "y": 0.92},
    "right_ankle": {"x": 0.65, "y": 0.92},
    "left_heel": {"x": 0.35, "y": 0.96},
    "right_heel": {"x": 0.66, "y": 0.96},
    "left_foot_index": {"x": 0.30, "y": 0.97},
    "right_foot_index": {"x": 0.72, "y": 0.97},
    "nose": {"x": 0.50, "y": 0.12},
}


VALID_PAYLOAD = {
    "scene": {
        "scene_type": "outdoor",
        "location_category": "park",
        "lighting": "natural",
        "mood": "casual",
        "space_constraints": {
            "available_width": "wide",
            "available_height": "open_sky",
            "ground_type": "flat",
        },
        "key_elements": ["greenery"],
        "composition_notes": "Open background with natural light.",
    },
    "recommendations": [
        {
            "name": "Open Scenic Pose",
            "category": "standing",
            "description": "Open both arms slightly and turn shoulders toward the light.",
            "reasoning": "The wide scene supports a relaxed open silhouette.",
            "keypoint_adjustments": {"arms": "open softly"},
            "difficulty": "easy",
            "camera_angle_suggestion": "eye-level",
            "marker_template": "arms_open",
            "marker_side": "auto",
            "marker_intensity": 1.0,
            "target_keypoints": TARGET_KEYPOINTS,
            "target_pose_quality_notes": "Arms and right hand are visibly changed from the current pose.",
        }
    ],
}

VALID_GUIDE_PAYLOAD = {
    "scene": VALID_PAYLOAD["scene"],
    "recommendation": {
        "pose_template_id": "seated_open_shoulders",
        "name": "Open Seated Shoulders",
        "category": "sitting",
        "description": "Open your shoulders and keep both hands relaxed.",
        "reasoning": "The subject is seated close to the camera, so a subtle upper-body guide is practical.",
        "difficulty": "easy",
        "camera_angle_suggestion": "eye-level",
        "guide_params": {"lean": 0.0},
        "correction_focus": ["shoulders", "wrists"],
    },
    "usage_notes": "Selector output only.",
}


class SchemaTests(unittest.TestCase):
    def test_valid_payload_is_accepted(self) -> None:
        parsed = VLMReasoningResult.model_validate(VALID_PAYLOAD)
        self.assertEqual(parsed.scene.location_category, "park")
        self.assertEqual(parsed.recommendations[0].marker_template, "arms_open")

    def test_missing_required_field_is_rejected(self) -> None:
        payload = dict(VALID_PAYLOAD)
        payload["scene"] = dict(VALID_PAYLOAD["scene"])
        del payload["scene"]["lighting"]
        with self.assertRaises(ValidationError):
            VLMReasoningResult.model_validate(payload)

    def test_missing_required_target_joint_is_rejected(self) -> None:
        payload = dict(VALID_PAYLOAD)
        payload["recommendations"] = [dict(VALID_PAYLOAD["recommendations"][0])]
        payload["recommendations"][0]["target_keypoints"] = dict(TARGET_KEYPOINTS)
        del payload["recommendations"][0]["target_keypoints"]["left_wrist"]
        with self.assertRaises(ValidationError):
            VLMReasoningResult.model_validate(payload)

    def test_out_of_range_target_joint_is_rejected(self) -> None:
        payload = dict(VALID_PAYLOAD)
        payload["recommendations"] = [dict(VALID_PAYLOAD["recommendations"][0])]
        payload["recommendations"][0]["target_keypoints"] = dict(TARGET_KEYPOINTS)
        payload["recommendations"][0]["target_keypoints"]["left_wrist"] = {"x": 1.2, "y": 0.48}
        with self.assertRaises(ValidationError):
            VLMReasoningResult.model_validate(payload)

    def test_valid_pose_guide_selector_payload_is_accepted(self) -> None:
        parsed = VLMPoseGuideResult.model_validate(VALID_GUIDE_PAYLOAD)
        self.assertEqual(parsed.recommendation.pose_template_id, "seated_open_shoulders")

    def test_pose_guide_missing_template_is_rejected(self) -> None:
        payload = dict(VALID_GUIDE_PAYLOAD)
        payload["recommendation"] = dict(VALID_GUIDE_PAYLOAD["recommendation"])
        del payload["recommendation"]["pose_template_id"]
        with self.assertRaises(ValidationError):
            VLMPoseGuideResult.model_validate(payload)

    def test_pose_guide_unknown_template_is_rejected(self) -> None:
        payload = dict(VALID_GUIDE_PAYLOAD)
        payload["recommendation"] = dict(VALID_GUIDE_PAYLOAD["recommendation"])
        payload["recommendation"]["pose_template_id"] = "flying_pose"
        with self.assertRaises(ValidationError):
            VLMPoseGuideResult.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
