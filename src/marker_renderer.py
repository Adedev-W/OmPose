from __future__ import annotations

from src.models.schemas import PoseData, PoseRecommendation


BODY_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
]

BODY_JOINTS = sorted({joint for connection in BODY_CONNECTIONS for joint in connection})


class MarkerRenderer:
    def __init__(
        self,
        target_color_bgr: tuple[int, int, int] = (0, 255, 128),
        current_color_bgr: tuple[int, int, int] = (185, 185, 185),
        target_alpha: float = 0.72,
        current_alpha: float = 0.34,
    ) -> None:
        self.target_color_bgr = target_color_bgr
        self.current_color_bgr = current_color_bgr
        self.target_alpha = target_alpha
        self.current_alpha = current_alpha

    def render(self, image_bgr, current_pose: PoseData, recommendation: PoseRecommendation):
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise RuntimeError("opencv-python-headless is required for marker rendering") from exc

        result = image_bgr.copy()
        current_layer = image_bgr.copy()
        target_layer = image_bgr.copy()

        self._draw_skeleton(
            current_layer,
            self.current_pose_points(current_pose),
            color=self.current_color_bgr,
            line_thickness=2,
            joint_radius=4,
        )
        result = cv2.addWeighted(current_layer, self.current_alpha, result, 1.0 - self.current_alpha, 0)

        self._draw_skeleton(
            target_layer,
            self.target_points(current_pose.image_width, current_pose.image_height, recommendation),
            color=self.target_color_bgr,
            line_thickness=5,
            joint_radius=7,
        )
        return cv2.addWeighted(target_layer, self.target_alpha, result, 1.0 - self.target_alpha, 0)

    def render_contact_sheet(
        self,
        image_bgr,
        current_pose: PoseData,
        recommendations: list[PoseRecommendation],
        max_recommendations: int = 3,
    ):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise RuntimeError("opencv-python-headless and numpy are required for contact sheets") from exc

        panels = [self.render_current_pose_panel(image_bgr, current_pose)]
        panels.extend(self.render(image_bgr, current_pose, item) for item in recommendations[:max_recommendations])

        panel_width = 360
        panel_height = int(round(panel_width * image_bgr.shape[0] / image_bgr.shape[1]))
        label_height = 42
        rendered_panels = []
        labels = ["Current pose"]
        labels.extend(f"{index + 1}. {item.name} ({item.difficulty})" for index, item in enumerate(recommendations[:max_recommendations]))

        for panel, label in zip(panels, labels, strict=True):
            resized = cv2.resize(panel, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
            canvas = np.full((panel_height + label_height, panel_width, 3), 245, dtype=np.uint8)
            canvas[:panel_height, :] = resized
            cv2.putText(
                canvas,
                self._truncate_label(label, 38),
                (12, panel_height + 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (35, 35, 35),
                1,
                cv2.LINE_AA,
            )
            rendered_panels.append(canvas)

        return np.hstack(rendered_panels)

    def render_current_pose_panel(self, image_bgr, current_pose: PoseData):
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise RuntimeError("opencv-python-headless is required for marker rendering") from exc

        layer = image_bgr.copy()
        self._draw_skeleton(
            layer,
            self.current_pose_points(current_pose),
            color=self.current_color_bgr,
            line_thickness=4,
            joint_radius=6,
        )
        return cv2.addWeighted(layer, 0.58, image_bgr, 0.42, 0)

    def render_stream_overlay(
        self,
        image_bgr,
        current_pose: PoseData,
        recommendation: PoseRecommendation | None = None,
    ):
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise RuntimeError("opencv-python-headless is required for marker rendering") from exc

        result = image_bgr.copy()
        current_layer = image_bgr.copy()
        self._draw_skeleton(
            current_layer,
            self.current_pose_points(current_pose),
            color=self.current_color_bgr,
            line_thickness=2,
            joint_radius=4,
        )
        result = cv2.addWeighted(current_layer, 0.46, result, 0.54, 0)

        if recommendation is not None:
            target_layer = image_bgr.copy()
            self._draw_skeleton(
                target_layer,
                self.target_points(current_pose.image_width, current_pose.image_height, recommendation),
                color=self.target_color_bgr,
                line_thickness=4,
                joint_radius=6,
            )
            result = cv2.addWeighted(target_layer, 0.66, result, 0.34, 0)
        return result

    def encode_jpeg(self, image_bgr, quality: int = 75) -> bytes:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise RuntimeError("opencv-python-headless is required for JPEG encoding") from exc

        quality = max(1, min(100, int(quality)))
        ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("failed to encode overlay as JPEG")
        return encoded.tobytes()

    def target_points(
        self,
        image_width: int,
        image_height: int,
        recommendation: PoseRecommendation,
    ) -> dict[str, tuple[int, int]]:
        points = {}
        for name in BODY_JOINTS:
            keypoint = recommendation.target_keypoints[name]
            points[name] = (
                int(round(keypoint.x * (image_width - 1))),
                int(round(keypoint.y * (image_height - 1))),
            )
        return points

    def current_pose_points(self, current_pose: PoseData) -> dict[str, tuple[int, int]]:
        return {
            name: (landmark.pixel_x, landmark.pixel_y)
            for name, landmark in current_pose.keypoints.items()
            if name in BODY_JOINTS
        }

    def _draw_skeleton(
        self,
        image,
        points: dict[str, tuple[int, int]],
        color: tuple[int, int, int],
        line_thickness: int,
        joint_radius: int,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise RuntimeError("opencv-python-headless is required for marker rendering") from exc

        for start, end in BODY_CONNECTIONS:
            if start not in points or end not in points:
                continue
            cv2.line(image, points[start], points[end], color, thickness=line_thickness, lineType=cv2.LINE_AA)

        for name in BODY_JOINTS:
            if name not in points:
                continue
            cv2.circle(image, points[name], joint_radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(image, points[name], joint_radius, color, 2, lineType=cv2.LINE_AA)

    def _truncate_label(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3].rstrip() + "..."
