from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.models.schemas import PoseRecommendation


class StreamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "config"
    return_overlay: bool = True
    jpeg_quality: int = Field(default=75, ge=1, le=100)
    max_fps: float = Field(default=10.0, gt=0.0, le=60.0)
    imgsz: int | None = Field(default=None, ge=160, le=1280)
    target_pose: PoseRecommendation | None = None


class TargetPoseMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "target_pose"
    target_pose: PoseRecommendation | None = None

