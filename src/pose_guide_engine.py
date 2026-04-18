from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.models.schemas import (
    MarkerTemplate,
    PoseData,
    PoseGuideRecommendation,
    PoseRecommendation,
    PoseTemplateId,
    TargetKeypoint,
)


SEATED_SAFE_TEMPLATES: dict[PoseTemplateId, PoseTemplateId] = {
    "standing_relaxed": "seated_relaxed",
    "standing_contrapposto": "seated_side_lean",
    "walking_stride_soft": "seated_open_shoulders",
    "arms_open_soft": "seated_open_shoulders",
    "hand_on_hip": "seated_hand_on_thigh",
    "leaning_side": "seated_side_lean",
}

TEMPLATE_TO_MARKER: dict[PoseTemplateId, MarkerTemplate] = {
    "seated_relaxed": "seated_lean",
    "seated_open_shoulders": "seated_lean",
    "seated_hand_on_thigh": "hand_on_hip",
    "seated_side_lean": "leaning_side",
    "standing_relaxed": "contrapposto",
    "standing_contrapposto": "contrapposto",
    "hand_on_hip": "hand_on_hip",
    "arms_open_soft": "arms_open",
    "walking_stride_soft": "walking_stride",
    "leaning_side": "leaning_side",
    "portrait_chin_angle": "leaning_side",
    "one_hand_near_face": "hand_on_hip",
}


@dataclass(frozen=True)
class BodyFrame:
    center_x: float
    shoulder_y: float
    hip_y: float
    shoulder_width: float
    hip_width: float
    torso_height: float
    left_sign: float
    upper_arm: float
    lower_arm: float
    upper_leg: float
    lower_leg: float
    seated_like: bool


class PoseGuideEngine:
    def generate(self, current_pose: PoseData, guide: PoseGuideRecommendation) -> PoseRecommendation:
        frame = self._body_frame(current_pose)
        template_id = self._effective_template(guide.pose_template_id, frame)
        target = self._build_points(current_pose, frame, template_id, guide.guide_params)
        target = self._fill_required_points(target, current_pose, frame, template_id)
        target = {name: TargetKeypoint(x=self._clamp(point.x), y=self._clamp(point.y), visibility=point.visibility, note=point.note) for name, point in target.items()}

        return PoseRecommendation(
            name=guide.name,
            category=self._category_for_template(template_id, guide.category),
            description=guide.description,
            reasoning=guide.reasoning,
            keypoint_adjustments=self._keypoint_adjustments(template_id),
            difficulty=guide.difficulty,
            camera_angle_suggestion=guide.camera_angle_suggestion,
            marker_template=TEMPLATE_TO_MARKER[template_id],
            marker_side="auto",
            marker_intensity=1.0,
            target_keypoints=target,
            target_pose_quality_notes=f"Generated locally from {template_id} using current body scale.",
            pose_template_id=template_id,
            guide_params=guide.guide_params,
            correction_focus=guide.correction_focus,
        )

    def _effective_template(self, template_id: PoseTemplateId, frame: BodyFrame) -> PoseTemplateId:
        if frame.seated_like:
            return SEATED_SAFE_TEMPLATES.get(template_id, template_id)
        return template_id

    def _build_points(
        self,
        current_pose: PoseData,
        frame: BodyFrame,
        template_id: PoseTemplateId,
        guide_params: dict[str, Any],
    ) -> dict[str, TargetKeypoint]:
        lean = self._guide_float(guide_params, "lean", default=0.0, low=-1.0, high=1.0)
        if template_id in {"seated_side_lean", "leaning_side", "portrait_chin_angle"} and lean == 0.0:
            lean = -0.45
        if template_id in {"standing_contrapposto"} and lean == 0.0:
            lean = 0.35

        torso_dx = lean * frame.shoulder_width * 0.18
        shoulder_center_x = frame.center_x + torso_dx
        hip_center_x = frame.center_x - torso_dx * 0.35
        shoulder_y = frame.shoulder_y + abs(lean) * frame.torso_height * 0.04
        hip_y = frame.hip_y

        points = {
            "left_shoulder": self._point(shoulder_center_x + frame.left_sign * frame.shoulder_width / 2.0, shoulder_y),
            "right_shoulder": self._point(shoulder_center_x - frame.left_sign * frame.shoulder_width / 2.0, shoulder_y),
            "left_hip": self._point(hip_center_x + frame.left_sign * frame.hip_width / 2.0, hip_y),
            "right_hip": self._point(hip_center_x - frame.left_sign * frame.hip_width / 2.0, hip_y),
        }
        points.update(self._arm_points(points, frame, template_id))
        points.update(self._leg_points(points, current_pose, frame, template_id))

        nose = current_pose.keypoints.get("nose")
        if nose is not None:
            head_shift = lean * frame.shoulder_width * 0.08
            if template_id == "portrait_chin_angle":
                head_shift += frame.left_sign * frame.shoulder_width * 0.08
            points["nose"] = self._point(nose.x + head_shift, max(0.04, nose.y - frame.torso_height * 0.03))
        return points

    def _arm_points(
        self,
        points: dict[str, TargetKeypoint],
        frame: BodyFrame,
        template_id: PoseTemplateId,
    ) -> dict[str, TargetKeypoint]:
        left_shoulder = points["left_shoulder"]
        right_shoulder = points["right_shoulder"]
        left_hip = points["left_hip"]
        right_hip = points["right_hip"]
        down = frame.torso_height
        outward = frame.left_sign

        if template_id in {"seated_open_shoulders", "arms_open_soft"}:
            return {
                "left_elbow": self._point(left_shoulder.x + outward * frame.upper_arm * 0.55, left_shoulder.y + down * 0.28),
                "left_wrist": self._point(left_shoulder.x + outward * (frame.upper_arm + frame.lower_arm) * 0.55, left_shoulder.y + down * 0.48),
                "right_elbow": self._point(right_shoulder.x - outward * frame.upper_arm * 0.55, right_shoulder.y + down * 0.28),
                "right_wrist": self._point(right_shoulder.x - outward * (frame.upper_arm + frame.lower_arm) * 0.55, right_shoulder.y + down * 0.48),
            }

        if template_id in {"seated_hand_on_thigh", "hand_on_hip"}:
            return {
                "left_elbow": self._point(left_shoulder.x + outward * frame.upper_arm * 0.25, left_shoulder.y + down * 0.38),
                "left_wrist": self._point(left_hip.x + outward * frame.hip_width * 0.35, left_hip.y + down * 0.10),
                "right_elbow": self._point(right_shoulder.x - outward * frame.upper_arm * 0.10, right_shoulder.y + down * 0.36),
                "right_wrist": self._point(right_hip.x, right_hip.y + down * 0.18),
            }

        if template_id == "one_hand_near_face":
            face_y = max(0.08, left_shoulder.y - down * 0.28)
            return {
                "left_elbow": self._point(left_shoulder.x + outward * frame.upper_arm * 0.18, left_shoulder.y + down * 0.18),
                "left_wrist": self._point(left_shoulder.x - outward * frame.shoulder_width * 0.12, face_y),
                "right_elbow": self._point(right_shoulder.x - outward * frame.upper_arm * 0.15, right_shoulder.y + down * 0.34),
                "right_wrist": self._point(right_hip.x, right_hip.y + down * 0.16),
            }

        if template_id == "walking_stride_soft":
            return {
                "left_elbow": self._point(left_shoulder.x - outward * frame.upper_arm * 0.12, left_shoulder.y + down * 0.34),
                "left_wrist": self._point(left_hip.x - outward * frame.shoulder_width * 0.15, left_hip.y + down * 0.02),
                "right_elbow": self._point(right_shoulder.x + outward * frame.upper_arm * 0.14, right_shoulder.y + down * 0.28),
                "right_wrist": self._point(right_shoulder.x + outward * frame.shoulder_width * 0.24, right_shoulder.y + down * 0.54),
            }

        return {
            "left_elbow": self._point(left_shoulder.x + outward * frame.upper_arm * 0.18, left_shoulder.y + down * 0.36),
            "left_wrist": self._point(left_hip.x + outward * frame.hip_width * 0.28, left_hip.y + down * 0.18),
            "right_elbow": self._point(right_shoulder.x - outward * frame.upper_arm * 0.18, right_shoulder.y + down * 0.36),
            "right_wrist": self._point(right_hip.x - outward * frame.hip_width * 0.28, right_hip.y + down * 0.18),
        }

    def _leg_points(
        self,
        points: dict[str, TargetKeypoint],
        current_pose: PoseData,
        frame: BodyFrame,
        template_id: PoseTemplateId,
    ) -> dict[str, TargetKeypoint]:
        left_hip = points["left_hip"]
        right_hip = points["right_hip"]
        outward = frame.left_sign
        seated = frame.seated_like or template_id.startswith("seated")

        if seated:
            knee_y = min(0.95, max(self._current_y(current_pose, "left_knee", frame.hip_y + frame.torso_height * 0.42), frame.hip_y + frame.torso_height * 0.24))
            ankle_y = min(0.985, max(self._current_y(current_pose, "left_ankle", knee_y + frame.torso_height * 0.26), knee_y + frame.torso_height * 0.08))
            return {
                "left_knee": self._point(left_hip.x + outward * frame.hip_width * 0.34, knee_y),
                "right_knee": self._point(right_hip.x - outward * frame.hip_width * 0.34, knee_y),
                "left_ankle": self._point(left_hip.x + outward * frame.hip_width * 0.28, ankle_y),
                "right_ankle": self._point(right_hip.x - outward * frame.hip_width * 0.28, ankle_y),
            }

        stride = frame.hip_width * 0.30 if template_id == "walking_stride_soft" else 0.0
        return {
            "left_knee": self._point(left_hip.x + outward * (frame.hip_width * 0.12 + stride), left_hip.y + frame.upper_leg),
            "right_knee": self._point(right_hip.x - outward * (frame.hip_width * 0.12 + stride * 0.25), right_hip.y + frame.upper_leg * 0.94),
            "left_ankle": self._point(left_hip.x + outward * (frame.hip_width * 0.22 + stride), left_hip.y + frame.upper_leg + frame.lower_leg),
            "right_ankle": self._point(right_hip.x - outward * (frame.hip_width * 0.18 - stride * 0.25), right_hip.y + frame.upper_leg + frame.lower_leg * 0.96),
        }

    def _fill_required_points(
        self,
        points: dict[str, TargetKeypoint],
        current_pose: PoseData,
        frame: BodyFrame,
        template_id: PoseTemplateId,
    ) -> dict[str, TargetKeypoint]:
        for side in ("left", "right"):
            sign = frame.left_sign if side == "left" else -frame.left_sign
            ankle = points[f"{side}_ankle"]
            current_heel = current_pose.keypoints.get(f"{side}_heel")
            current_foot = current_pose.keypoints.get(f"{side}_foot_index")
            points[f"{side}_heel"] = self._point(
                current_heel.x if current_heel else ankle.x - sign * frame.hip_width * 0.06,
                current_heel.y if current_heel else min(0.995, ankle.y + frame.torso_height * 0.05),
            )
            points[f"{side}_foot_index"] = self._point(
                current_foot.x if current_foot else ankle.x + sign * frame.hip_width * 0.16,
                current_foot.y if current_foot else min(0.995, ankle.y + frame.torso_height * 0.06),
            )
        return points

    def _body_frame(self, pose: PoseData) -> BodyFrame:
        left_shoulder = pose.keypoints.get("left_shoulder")
        right_shoulder = pose.keypoints.get("right_shoulder")
        left_hip = pose.keypoints.get("left_hip")
        right_hip = pose.keypoints.get("right_hip")

        shoulder_center_x = self._mid_x(left_shoulder, right_shoulder, 0.50)
        shoulder_y = self._mid_y(left_shoulder, right_shoulder, 0.34)
        hip_center_x = self._mid_x(left_hip, right_hip, shoulder_center_x)
        hip_y = self._mid_y(left_hip, right_hip, shoulder_y + 0.34)
        center_x = (shoulder_center_x + hip_center_x) / 2.0

        shoulder_width = self._distance_landmarks(left_shoulder, right_shoulder, 0.24)
        hip_width = self._distance_landmarks(left_hip, right_hip, shoulder_width * 0.72)
        torso_height = max(0.14, min(0.46, abs(hip_y - shoulder_y)))
        shoulder_width = max(0.12, min(0.58, shoulder_width))
        hip_width = max(0.10, min(0.48, hip_width))
        left_sign = 1.0
        if left_shoulder is not None and right_shoulder is not None and left_shoulder.x < right_shoulder.x:
            left_sign = -1.0

        upper_arm = self._segment(pose, "left_shoulder", "left_elbow", shoulder_width * 0.52)
        lower_arm = self._segment(pose, "left_elbow", "left_wrist", shoulder_width * 0.55)
        upper_leg = self._segment(pose, "left_hip", "left_knee", torso_height * 0.55)
        lower_leg = self._segment(pose, "left_knee", "left_ankle", torso_height * 0.62)

        hip_or_knee_low = hip_y > 0.66 or self._current_y(pose, "left_knee", 0.0) > 0.82
        body_large = torso_height > 0.32 or shoulder_width > 0.42
        seated_like = hip_or_knee_low or body_large
        return BodyFrame(
            center_x=center_x,
            shoulder_y=shoulder_y,
            hip_y=hip_y,
            shoulder_width=shoulder_width,
            hip_width=hip_width,
            torso_height=torso_height,
            left_sign=left_sign,
            upper_arm=max(0.09, min(0.32, upper_arm)),
            lower_arm=max(0.08, min(0.30, lower_arm)),
            upper_leg=max(0.10, min(0.34, upper_leg)),
            lower_leg=max(0.10, min(0.36, lower_leg)),
            seated_like=seated_like,
        )

    def _keypoint_adjustments(self, template_id: PoseTemplateId) -> dict[str, str]:
        if template_id in {"seated_open_shoulders", "arms_open_soft"}:
            return {"arms": "open elbows outward softly", "torso": "keep chest open", "head": "keep face visible"}
        if template_id in {"seated_hand_on_thigh", "hand_on_hip"}:
            return {"arms": "place one hand near thigh or hip", "torso": "keep shoulders relaxed"}
        if template_id in {"seated_side_lean", "leaning_side"}:
            return {"torso": "lean slightly to one side", "arms": "keep hands relaxed"}
        if template_id == "one_hand_near_face":
            return {"arms": "bring one hand near face without covering it", "head": "keep chin open"}
        return {"torso": "keep posture clean", "arms": "relax arms with visible silhouette"}

    def _category_for_template(self, template_id: PoseTemplateId, fallback: str) -> str:
        if template_id.startswith("seated"):
            return "sitting"
        if template_id == "walking_stride_soft":
            return "walking"
        if template_id in {"leaning_side", "portrait_chin_angle"}:
            return "leaning"
        if template_id in {"standing_relaxed", "standing_contrapposto", "hand_on_hip", "arms_open_soft"}:
            return "standing"
        return fallback

    def _point(self, x: float, y: float) -> TargetKeypoint:
        return TargetKeypoint(x=self._clamp(x), y=self._clamp(y), visibility=1.0)

    def _clamp(self, value: float) -> float:
        return max(0.01, min(0.99, float(value)))

    def _guide_float(self, guide_params: dict[str, Any], name: str, default: float, low: float, high: float) -> float:
        value = guide_params.get(name, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return max(low, min(high, parsed))

    def _mid_x(self, first, second, default: float) -> float:
        if first is not None and second is not None:
            return (first.x + second.x) / 2.0
        return default

    def _mid_y(self, first, second, default: float) -> float:
        if first is not None and second is not None:
            return (first.y + second.y) / 2.0
        return default

    def _distance_landmarks(self, first, second, default: float) -> float:
        if first is None or second is None:
            return default
        return math.hypot(first.x - second.x, first.y - second.y)

    def _segment(self, pose: PoseData, start: str, end: str, default: float) -> float:
        return self._distance_landmarks(pose.keypoints.get(start), pose.keypoints.get(end), default)

    def _current_y(self, pose: PoseData, name: str, default: float) -> float:
        landmark = pose.keypoints.get(name)
        return landmark.y if landmark is not None else default
