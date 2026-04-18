from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from src.errors import ConfigError


DEFAULT_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
DEFAULT_POSE_MODEL_PATH = Path("assets/models/pose_landmarker_heavy.task")


def ensure_pose_model_asset(
    model_task_path: Path | None = None,
    model_url: str = DEFAULT_POSE_MODEL_URL,
) -> Path:
    target = model_task_path or DEFAULT_POSE_MODEL_PATH
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    try:
        with urllib.request.urlopen(model_url, timeout=120) as response:
            with tmp_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        tmp_path.replace(target)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise ConfigError(
            "failed to download MediaPipe pose model asset. "
            f"Provide --model-task-path or manually place the file at {target}."
        ) from exc
    return target

