from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MarkerTemplate = Literal[
    "contrapposto",
    "hand_on_hip",
    "arms_open",
    "walking_stride",
    "leaning_side",
    "seated_lean",
]
MarkerSide = Literal["left", "right", "center", "auto"]
PoseCategory = Literal["standing", "sitting", "leaning", "walking", "dramatic", "casual"]
Difficulty = Literal["easy", "medium", "hard"]
KeypointProfile = Literal["mediapipe33", "coco17", "unknown"]
PoseTemplateId = Literal[
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
]

REQUIRED_TARGET_KEYPOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

TARGET_SEGMENTS = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
)


class PoseLandmark(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0, le=32)
    name: str
    x: float
    y: float
    z: float | None = None
    pixel_x: int
    pixel_y: int
    visibility: float | None = Field(default=None, ge=0.0, le=1.0)
    presence: float | None = Field(default=None, ge=0.0, le=1.0)
    world_x: float | None = None
    world_y: float | None = None
    world_z: float | None = None


class PoseData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    keypoint_profile: KeypointProfile = "mediapipe33"
    keypoints: dict[str, PoseLandmark]
    confidence: float = Field(ge=0.0, le=1.0)
    has_world_landmarks: bool = False

    def compact_summary(self) -> dict[str, Any]:
        names = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "confidence": round(self.confidence, 4),
            "keypoints": {
                name: {
                    "x": round(self.keypoints[name].x, 4),
                    "y": round(self.keypoints[name].y, 4),
                    "visibility": self.keypoints[name].visibility,
                }
                for name in names
                if name in self.keypoints
            },
        }


class SpaceConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available_width: Literal["narrow", "medium", "wide"]
    available_height: Literal["low_ceiling", "medium", "open_sky"]
    ground_type: Literal["flat", "stairs", "uneven", "seated_available", "unknown"]


class SceneContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_type: Literal["indoor", "outdoor", "mixed", "unknown"]
    location_category: str = Field(min_length=1)
    lighting: str = Field(min_length=1)
    mood: str = Field(min_length=1)
    space_constraints: SpaceConstraints
    key_elements: list[str] = Field(default_factory=list)
    composition_notes: str = Field(min_length=1)


class TargetKeypoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    visibility: float | None = Field(default=None, ge=0.0, le=1.0)
    note: str | None = None


class PoseRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    category: PoseCategory
    description: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    keypoint_adjustments: dict[str, str] = Field(default_factory=dict)
    difficulty: Difficulty
    camera_angle_suggestion: str = Field(min_length=1)
    marker_template: MarkerTemplate
    marker_side: MarkerSide = "auto"
    marker_intensity: float = Field(default=1.0, ge=0.25, le=1.5)
    target_keypoints: dict[str, TargetKeypoint]
    target_pose_quality_notes: str | None = None
    pose_template_id: PoseTemplateId | None = None
    guide_params: dict[str, Any] = Field(default_factory=dict)
    correction_focus: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_target_keypoints(self) -> "PoseRecommendation":
        missing = [name for name in REQUIRED_TARGET_KEYPOINTS if name not in self.target_keypoints]
        if missing:
            raise ValueError(f"target_keypoints missing required joints: {', '.join(missing)}")

        shoulder_width = _distance(self.target_keypoints, "left_shoulder", "right_shoulder")
        torso_height = _midpoint_distance(
            self.target_keypoints,
            ("left_shoulder", "right_shoulder"),
            ("left_hip", "right_hip"),
        )
        if shoulder_width < 0.03 or shoulder_width > 0.75:
            raise ValueError("target_keypoints shoulder width is implausible")
        if torso_height < 0.04 or torso_height > 0.75:
            raise ValueError("target_keypoints torso height is implausible")

        for start, end in TARGET_SEGMENTS:
            segment = _distance(self.target_keypoints, start, end)
            if segment < 0.01 or segment > 0.85:
                raise ValueError(f"target_keypoints segment {start}->{end} is implausible")
        return self


class VLMReasoningResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene: SceneContext
    recommendations: list[PoseRecommendation] = Field(min_length=1)

    @field_validator("recommendations")
    @classmethod
    def limit_recommendations(cls, value: list[PoseRecommendation]) -> list[PoseRecommendation]:
        if len(value) > 10:
            raise ValueError("recommendations must contain at most 10 items")
        return value


class PoseGuideRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pose_template_id: PoseTemplateId
    name: str = Field(min_length=1)
    category: PoseCategory
    description: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    difficulty: Difficulty
    camera_angle_suggestion: str = Field(min_length=1)
    guide_params: dict[str, Any] = Field(default_factory=dict)
    correction_focus: list[str] = Field(default_factory=list)


class VLMPoseGuideResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene: SceneContext
    recommendation: PoseGuideRecommendation
    usage_notes: str | None = None


class PipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_image: str
    overlay_image: str
    contact_sheet_image: str
    result_json: str
    scene: SceneContext
    current_pose: PoseData
    recommendations: list[PoseRecommendation]
    selected_recommendation_index: int = 0
    usage: dict[str, Any] = Field(default_factory=dict)


class PoseGuideResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_image: str
    scene: SceneContext
    current_pose: PoseData
    selected_pose_guide: PoseGuideRecommendation
    target_pose: PoseRecommendation
    usage: dict[str, Any] = Field(default_factory=dict)


def _distance(keypoints: dict[str, TargetKeypoint], start: str, end: str) -> float:
    first = keypoints[start]
    second = keypoints[end]
    return math.hypot(first.x - second.x, first.y - second.y)


def _midpoint_distance(
    keypoints: dict[str, TargetKeypoint],
    first_pair: tuple[str, str],
    second_pair: tuple[str, str],
) -> float:
    first_x = (keypoints[first_pair[0]].x + keypoints[first_pair[1]].x) / 2.0
    first_y = (keypoints[first_pair[0]].y + keypoints[first_pair[1]].y) / 2.0
    second_x = (keypoints[second_pair[0]].x + keypoints[second_pair[1]].x) / 2.0
    second_y = (keypoints[second_pair[0]].y + keypoints[second_pair[1]].y) / 2.0
    return math.hypot(first_x - second_x, first_y - second_y)
