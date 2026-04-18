from __future__ import annotations

import os
from pathlib import Path

from src.errors import ImageInputError, NoPoseDetectedError
from src.model_assets import ensure_pose_model_asset
from src.models.schemas import PoseData, PoseLandmark


LANDMARK_NAMES = [
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


class PoseDetector:
    def __init__(
        self,
        model_task_path: Path | None = None,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_poses: int = 1,
    ) -> None:
        self.model_task_path = model_task_path
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_pose_presence_confidence = min_pose_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.num_poses = num_poses

    def detect(self, image_path: Path, pose_index: int = 0) -> PoseData:
        try:
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/ompose-matplotlib")
            import cv2
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise ImageInputError("mediapipe and opencv-python-headless are required for pose detection") from exc

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ImageInputError(f"failed to read image: {image_path}")

        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        model_path = ensure_pose_model_asset(self.model_task_path)

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        with PoseLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)

        if not result.pose_landmarks:
            raise NoPoseDetectedError("no person pose was detected in the image")
        if pose_index < 0 or pose_index >= len(result.pose_landmarks):
            raise NoPoseDetectedError(f"pose index {pose_index} is unavailable; detected {len(result.pose_landmarks)} pose(s)")

        landmarks = result.pose_landmarks[pose_index]
        world_landmarks = (
            result.pose_world_landmarks[pose_index]
            if result.pose_world_landmarks and pose_index < len(result.pose_world_landmarks)
            else None
        )
        keypoints: dict[str, PoseLandmark] = {}
        visibility_values: list[float] = []

        for index, landmark in enumerate(landmarks):
            name = LANDMARK_NAMES[index]
            visibility = getattr(landmark, "visibility", None)
            presence = getattr(landmark, "presence", None)
            if visibility is not None:
                visibility_values.append(float(visibility))
            world = world_landmarks[index] if world_landmarks else None
            keypoints[name] = PoseLandmark(
                index=index,
                name=name,
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(getattr(landmark, "z", 0.0)),
                pixel_x=int(round(float(landmark.x) * width)),
                pixel_y=int(round(float(landmark.y) * height)),
                visibility=float(visibility) if visibility is not None else None,
                presence=float(presence) if presence is not None else None,
                world_x=float(world.x) if world is not None else None,
                world_y=float(world.y) if world is not None else None,
                world_z=float(world.z) if world is not None else None,
            )

        confidence = sum(visibility_values) / len(visibility_values) if visibility_values else 0.0
        return PoseData(
            image_width=width,
            image_height=height,
            keypoint_profile="mediapipe33",
            keypoints=keypoints,
            confidence=max(0.0, min(1.0, confidence)),
            has_world_landmarks=world_landmarks is not None,
        )
