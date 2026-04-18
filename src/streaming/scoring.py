from __future__ import annotations

import math

from src.models.schemas import PoseData, PoseRecommendation


SCORING_JOINTS = (
    "nose",
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
)


def score_pose_alignment(current_pose: PoseData, target_pose: PoseRecommendation | None) -> tuple[float | None, list[str]]:
    if target_pose is None:
        return None, []

    distances = []
    corrections = []
    for joint in SCORING_JOINTS:
        current = current_pose.keypoints.get(joint)
        target = target_pose.target_keypoints.get(joint)
        if current is None or target is None:
            continue
        dx = target.x - current.x
        dy = target.y - current.y
        distance = math.hypot(dx, dy)
        distances.append(distance)
        if distance >= 0.055:
            corrections.append(_correction_for_delta(joint, dx, dy))

    if not distances:
        return None, []

    mean_distance = sum(distances) / len(distances)
    score = max(0.0, min(1.0, 1.0 - (mean_distance / 0.28)))
    return round(score, 4), corrections[:5]


def _correction_for_delta(joint: str, dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        direction = "right" if dx > 0 else "left"
        return f"move_{joint}_{direction}"
    direction = "down" if dy > 0 else "up"
    return f"move_{joint}_{direction}"

