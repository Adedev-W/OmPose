from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from src.api.schemas import StreamConfig, TargetPoseMessage
from src.errors import NoPoseDetectedError, OmPoseError
from src.marker_renderer import MarkerRenderer
from src.models.schemas import PoseData, PoseRecommendation
from src.streaming.scoring import score_pose_alignment


@dataclass
class FramePacket:
    frame_id: int
    data: bytes
    received_at: float
    dropped_frames: int


class LatestFrameBuffer:
    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._latest: FramePacket | None = None
        self._frame_id = 0
        self._dropped_frames = 0
        self.closed = False

    async def put(self, data: bytes) -> None:
        async with self._condition:
            if self._latest is not None:
                self._dropped_frames += 1
            self._frame_id += 1
            self._latest = FramePacket(
                frame_id=self._frame_id,
                data=data,
                received_at=time.perf_counter(),
                dropped_frames=self._dropped_frames,
            )
            self._condition.notify()

    async def get(self) -> FramePacket | None:
        async with self._condition:
            while self._latest is None and not self.closed:
                await self._condition.wait()
            packet = self._latest
            self._latest = None
            return packet

    async def close(self) -> None:
        async with self._condition:
            self.closed = True
            self._condition.notify_all()


class PoseSmoother:
    def __init__(self, alpha: float = 0.45, hold_seconds: float = 1.25) -> None:
        self.alpha = max(0.05, min(1.0, alpha))
        self.hold_seconds = max(0.0, hold_seconds)
        self._last_pose: PoseData | None = None
        self._last_seen_at = 0.0

    def update(self, pose: PoseData) -> PoseData:
        now = time.perf_counter()
        previous = self._last_pose
        if (
            previous is None
            or previous.image_width != pose.image_width
            or previous.image_height != pose.image_height
            or previous.keypoint_profile != pose.keypoint_profile
        ):
            self._last_pose = pose
            self._last_seen_at = now
            return pose

        smoothed_keypoints = {}
        for name, landmark in pose.keypoints.items():
            previous_landmark = previous.keypoints.get(name)
            if previous_landmark is None:
                smoothed_keypoints[name] = landmark
                continue

            x = self._mix(previous_landmark.x, landmark.x)
            y = self._mix(previous_landmark.y, landmark.y)
            visibility = self._mix_optional(previous_landmark.visibility, landmark.visibility)
            presence = self._mix_optional(previous_landmark.presence, landmark.presence)
            smoothed_keypoints[name] = landmark.model_copy(
                update={
                    "x": x,
                    "y": y,
                    "pixel_x": int(round(x * (pose.image_width - 1))),
                    "pixel_y": int(round(y * (pose.image_height - 1))),
                    "visibility": visibility,
                    "presence": presence,
                }
            )

        smoothed_pose = pose.model_copy(update={"keypoints": smoothed_keypoints})
        self._last_pose = smoothed_pose
        self._last_seen_at = now
        return smoothed_pose

    def held_pose(self) -> PoseData | None:
        if self._last_pose is None:
            return None
        if time.perf_counter() - self._last_seen_at > self.hold_seconds:
            return None
        return self._last_pose

    def _mix(self, previous: float, current: float) -> float:
        return (self.alpha * current) + ((1.0 - self.alpha) * previous)

    def _mix_optional(self, previous: float | None, current: float | None) -> float | None:
        if previous is None:
            return current
        if current is None:
            return previous
        return max(0.0, min(1.0, self._mix(previous, current)))


class StreamingPoseSession:
    def __init__(
        self,
        websocket,
        detector,
        renderer: MarkerRenderer | None = None,
        default_return_overlay: bool = True,
        default_max_fps: float = 10.0,
    ) -> None:
        self.websocket = websocket
        self.detector = detector
        self.renderer = renderer or MarkerRenderer()
        self.config = StreamConfig(return_overlay=default_return_overlay, max_fps=default_max_fps)
        self.target_pose: PoseRecommendation | None = None
        self.buffer = LatestFrameBuffer()
        self._last_sent_at = 0.0
        self.pose_smoother = PoseSmoother()

    async def run(self) -> None:
        await self.websocket.accept()
        try:
            await self._read_initial_config()
            receiver = asyncio.create_task(self._receive_loop())
            processor = asyncio.create_task(self._process_loop())
            done, pending = await asyncio.wait({receiver, processor}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
        finally:
            await self.buffer.close()

    async def _read_initial_config(self) -> None:
        message = await self.websocket.receive_text()
        try:
            self.config = StreamConfig.model_validate_json(message)
        except ValidationError as exc:
            await self.websocket.send_json({"type": "error", "message": f"invalid config: {exc}"})
            raise
        self.target_pose = self.config.target_pose
        await self.websocket.send_json(
            {
                "type": "ready",
                "return_overlay": self.config.return_overlay,
                "max_fps": self.config.max_fps,
            }
        )

    async def _receive_loop(self) -> None:
        while True:
            message = await self.websocket.receive()
            if message.get("bytes") is not None:
                await self.buffer.put(message["bytes"])
                continue
            if message.get("text") is not None:
                await self._handle_text_message(message["text"])

    async def _handle_text_message(self, text: str) -> None:
        data = json.loads(text)
        message_type = data.get("type")
        if message_type == "target_pose":
            parsed = TargetPoseMessage.model_validate(data)
            self.target_pose = parsed.target_pose
            await self.websocket.send_json({"type": "target_pose_updated", "has_target_pose": self.target_pose is not None})
        elif message_type == "config":
            parsed = StreamConfig.model_validate(data)
            self.config = parsed
            self.target_pose = parsed.target_pose if parsed.target_pose is not None else self.target_pose
            await self.websocket.send_json({"type": "config_updated"})
        else:
            await self.websocket.send_json({"type": "error", "message": f"unknown message type: {message_type}"})

    async def _process_loop(self) -> None:
        while True:
            packet = await self.buffer.get()
            if packet is None:
                return
            await self._throttle()
            result, overlay = await asyncio.to_thread(self._process_packet, packet)
            await self.websocket.send_json(result)
            if overlay is not None:
                await self.websocket.send_bytes(overlay)
            self._last_sent_at = time.perf_counter()

    async def _throttle(self) -> None:
        min_interval = 1.0 / self.config.max_fps
        elapsed = time.perf_counter() - self._last_sent_at
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)

    def _process_packet(self, packet: FramePacket) -> tuple[dict[str, Any], bytes | None]:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise RuntimeError("opencv-python-headless and numpy are required for streaming") from exc

        decoded = cv2.imdecode(np.frombuffer(packet.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            return (
                {
                    "type": "pose_result",
                    "frame_id": packet.frame_id,
                    "latency_ms": round((time.perf_counter() - packet.received_at) * 1000.0, 3),
                    "inference_ms": None,
                    "dropped_frames": packet.dropped_frames,
                    "current_pose": None,
                    "score": None,
                    "corrections": [],
                    "has_overlay": False,
                    "error": "invalid JPEG frame",
                },
                None,
            )

        inference_started = time.perf_counter()
        try:
            current_pose = self.pose_smoother.update(self.detector.detect_frame(decoded))
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            score, corrections = score_pose_alignment(current_pose, self.target_pose)
            overlay = None
            if self.config.return_overlay:
                overlay_image = self.renderer.render_stream_overlay(decoded, current_pose, self.target_pose)
                overlay = self.renderer.encode_jpeg(overlay_image, quality=self.config.jpeg_quality)
            return (
                {
                    "type": "pose_result",
                    "frame_id": packet.frame_id,
                    "latency_ms": round((time.perf_counter() - packet.received_at) * 1000.0, 3),
                    "inference_ms": round(inference_ms, 3),
                    "dropped_frames": packet.dropped_frames,
                    "current_pose": current_pose.model_dump(),
                    "score": score,
                    "corrections": corrections,
                    "has_overlay": overlay is not None,
                    "pose_stale": False,
                },
                overlay,
            )
        except NoPoseDetectedError as exc:
            held_pose = self.pose_smoother.held_pose()
            if held_pose is not None:
                score, corrections = score_pose_alignment(held_pose, self.target_pose)
                overlay = None
                if self.config.return_overlay:
                    overlay_image = self.renderer.render_stream_overlay(decoded, held_pose, self.target_pose)
                    overlay = self.renderer.encode_jpeg(overlay_image, quality=self.config.jpeg_quality)
                return (
                    {
                        "type": "pose_result",
                        "frame_id": packet.frame_id,
                        "latency_ms": round((time.perf_counter() - packet.received_at) * 1000.0, 3),
                        "inference_ms": round((time.perf_counter() - inference_started) * 1000.0, 3),
                        "dropped_frames": packet.dropped_frames,
                        "current_pose": held_pose.model_dump(),
                        "score": score,
                        "corrections": corrections,
                        "has_overlay": overlay is not None,
                        "pose_stale": True,
                        "warning": str(exc),
                    },
                    overlay,
                )
            return self._error_result(packet, inference_started, str(exc))
        except OmPoseError as exc:
            return self._error_result(packet, inference_started, str(exc))

    def _error_result(self, packet: FramePacket, inference_started: float, error: str) -> tuple[dict[str, Any], None]:
        return (
            {
                "type": "pose_result",
                "frame_id": packet.frame_id,
                "latency_ms": round((time.perf_counter() - packet.received_at) * 1000.0, 3),
                "inference_ms": round((time.perf_counter() - inference_started) * 1000.0, 3),
                "dropped_frames": packet.dropped_frames,
                "current_pose": None,
                "score": None,
                "corrections": [],
                "has_overlay": False,
                "pose_stale": False,
                "error": error,
            },
            None,
        )
