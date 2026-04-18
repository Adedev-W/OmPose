from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.errors import ImageInputError, NoPoseDetectedError
from src.models.schemas import PoseData, PoseLandmark


LOGGER = logging.getLogger(__name__)

DEFAULT_YOLO_MODEL_PATH = Path("assets/models/yolo26n-pose.pt")
COCO17_LANDMARK_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
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
]


@dataclass(frozen=True)
class YoloPoseSettings:
    model_path: Path = DEFAULT_YOLO_MODEL_PATH
    device: str = "cpu"
    imgsz: int = 320
    conf: float = 0.25
    max_det: int = 1
    torch_threads: int | None = None

    @classmethod
    def from_env(cls) -> "YoloPoseSettings":
        torch_threads = os.getenv("OMPOSE_YOLO_TORCH_THREADS")
        return cls(
            model_path=Path(os.getenv("OMPOSE_YOLO_MODEL_PATH", str(DEFAULT_YOLO_MODEL_PATH))),
            device=os.getenv("OMPOSE_YOLO_DEVICE", "cpu"),
            imgsz=int(os.getenv("OMPOSE_YOLO_IMGSZ", "320")),
            conf=float(os.getenv("OMPOSE_YOLO_CONF", "0.25")),
            max_det=int(os.getenv("OMPOSE_YOLO_MAX_DET", "1")),
            torch_threads=int(torch_threads) if torch_threads else None,
        )


class YoloPoseDetector:
    def __init__(self, settings: YoloPoseSettings | None = None, model: Any | None = None) -> None:
        self.settings = settings or YoloPoseSettings.from_env()
        self.model = model
        self.loaded = model is not None
        self._lock = threading.Lock()
        if self.settings.device == "cpu" and "x-pose" in self.settings.model_path.name:
            LOGGER.warning(
                "Using %s on CPU. Expect low streaming FPS; set OMPOSE_YOLO_MODEL_PATH to a smaller yolo26n/s-pose model for realtime.",
                self.settings.model_path,
            )

    def load(self) -> None:
        if self.loaded:
            return
        if not self.settings.model_path.exists():
            raise ImageInputError(f"YOLO pose model does not exist: {self.settings.model_path}")
        try:
            os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ompose-ultralytics")
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise ImageInputError("ultralytics is required for YOLO pose detection") from exc

        self._configure_cpu_runtime()
        self.model = YOLO(str(self.settings.model_path))
        if hasattr(self.model, "fuse"):
            try:
                self.model.fuse()
            except Exception as exc:  # pragma: no cover - optional optimization.
                LOGGER.debug("YOLO fuse skipped: %s", exc)
        self.loaded = True
        self.warmup()

    def warmup(self) -> None:
        if self.model is None:
            return
        try:
            import numpy as np

            warmup_frame = np.zeros((self.settings.imgsz, self.settings.imgsz, 3), dtype=np.uint8)
            self._predict(warmup_frame)
        except Exception as exc:  # pragma: no cover - warmup should never block startup.
            LOGGER.warning("YOLO warmup failed: %s", exc)

    def detect(self, image_path: Path, pose_index: int = 0) -> PoseData:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise ImageInputError("opencv-python-headless is required for YOLO image detection") from exc

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ImageInputError(f"failed to read image: {image_path}")
        return self.detect_frame(image_bgr, pose_index=pose_index)

    def detect_frame(self, image_bgr, pose_index: int = 0) -> PoseData:
        self.load()
        results = self._predict(image_bgr)
        return yolo_result_to_pose_data(results[0], image_bgr.shape[1], image_bgr.shape[0], pose_index=pose_index)

    def _predict(self, image_bgr):
        if self.model is None:
            raise ImageInputError("YOLO model is not loaded")
        with self._lock:
            return self.model.predict(
                source=image_bgr,
                device=self.settings.device,
                imgsz=self.settings.imgsz,
                conf=self.settings.conf,
                max_det=self.settings.max_det,
                stream=False,
                verbose=False,
            )

    def _configure_cpu_runtime(self) -> None:
        if self.settings.device != "cpu" or self.settings.torch_threads is None:
            return
        try:
            import torch

            threads = max(1, self.settings.torch_threads)
            torch.set_num_threads(threads)
            if hasattr(torch, "set_num_interop_threads"):
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    pass
        except Exception as exc:  # pragma: no cover - optimization guard.
            LOGGER.debug("Torch CPU runtime tuning skipped: %s", exc)

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "yolo26",
            "model_path": str(self.settings.model_path),
            "model_loaded": self.loaded,
            "device": self.settings.device,
            "imgsz": self.settings.imgsz,
            "conf": self.settings.conf,
            "max_det": self.settings.max_det,
            "torch_threads": self.settings.torch_threads,
            "keypoint_profile": "coco17",
        }


def yolo_result_to_pose_data(result: Any, image_width: int, image_height: int, pose_index: int = 0) -> PoseData:
    keypoints_obj = getattr(result, "keypoints", None)
    if keypoints_obj is None or getattr(keypoints_obj, "data", None) is None:
        raise NoPoseDetectedError("YOLO did not return pose keypoints")

    data = keypoints_obj.data
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()

    if len(data) == 0:
        raise NoPoseDetectedError("no person pose was detected in the image")
    if pose_index < 0 or pose_index >= len(data):
        raise NoPoseDetectedError(f"pose index {pose_index} is unavailable; detected {len(data)} pose(s)")

    selected = data[pose_index]
    keypoints: dict[str, PoseLandmark] = {}
    confidences: list[float] = []
    for index, name in enumerate(COCO17_LANDMARK_NAMES):
        values = selected[index]
        x_pixel = float(values[0])
        y_pixel = float(values[1])
        confidence = float(values[2]) if len(values) > 2 else None
        if confidence is not None:
            confidences.append(confidence)
        keypoints[name] = PoseLandmark(
            index=index,
            name=name,
            x=max(0.0, min(1.0, x_pixel / image_width)),
            y=max(0.0, min(1.0, y_pixel / image_height)),
            z=None,
            pixel_x=int(round(x_pixel)),
            pixel_y=int(round(y_pixel)),
            visibility=max(0.0, min(1.0, confidence)) if confidence is not None else None,
            presence=max(0.0, min(1.0, confidence)) if confidence is not None else None,
        )

    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return PoseData(
        image_width=image_width,
        image_height=image_height,
        keypoint_profile="coco17",
        keypoints=keypoints,
        confidence=max(0.0, min(1.0, confidence)),
        has_world_landmarks=False,
    )
