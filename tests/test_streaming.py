from __future__ import annotations

import asyncio
import json
import unittest

import cv2
import numpy as np

from src.api.app import create_app
from src.api.schemas import StreamConfig
from src.streaming.scoring import score_pose_alignment
from src.streaming.session import FramePacket, LatestFrameBuffer, StreamingPoseSession
from tests.test_marker_renderer import recommendation, synthetic_coco17_pose


class FakeStreamingDetector:
    def __init__(self) -> None:
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def detect_frame(self, image_bgr):
        return synthetic_coco17_pose(width=image_bgr.shape[1], height=image_bgr.shape[0])

    def metadata(self):
        return {
            "backend": "fake-yolo26",
            "model_path": "fake.pt",
            "model_loaded": self.loaded,
            "device": "cpu",
            "imgsz": 384,
            "conf": 0.35,
            "max_det": 1,
            "keypoint_profile": "coco17",
        }


class StreamingTests(unittest.TestCase):
    def test_latest_frame_buffer_drops_stale_frames(self) -> None:
        async def run():
            buffer = LatestFrameBuffer()
            await buffer.put(b"old")
            await buffer.put(b"new")
            packet = await buffer.get()
            self.assertEqual(packet.data, b"new")
            self.assertEqual(packet.dropped_frames, 1)

        asyncio.run(run())

    def test_pose_scoring_rewards_matching_target(self) -> None:
        pose = synthetic_coco17_pose()
        target = recommendation("arms_open")
        score, corrections = score_pose_alignment(pose, target)
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertTrue(corrections)

    def test_health_and_model_endpoints(self) -> None:
        app = create_app(detector=FakeStreamingDetector())
        routes = {route.path for route in app.routes}
        self.assertIn("/health", routes)
        self.assertIn("/models/current", routes)
        self.assertIn("/api/recommendations", routes)
        self.assertIn("/api/recommendations/upload", routes)
        self.assertIn("/api/recommendations/upload/async", routes)
        self.assertIn("/api/recommendations/jobs/{job_id}", routes)
        self.assertIn("/api/pose-guide/upload/async", routes)
        self.assertIn("/api/pose-guide/jobs/{job_id}", routes)
        self.assertIn("/ws/pose", routes)
        middleware_names = {middleware.cls.__name__ for middleware in app.user_middleware}
        self.assertIn("CORSMiddleware", middleware_names)

    def test_session_processing_returns_json_and_overlay(self) -> None:
        class FakeWebSocket:
            pass

        session = StreamingPoseSession(FakeWebSocket(), detector=FakeStreamingDetector())
        session.config = StreamConfig(return_overlay=True, jpeg_quality=70, max_fps=30)

        ok, encoded = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertTrue(ok)

        packet = FramePacket(frame_id=1, data=encoded.tobytes(), received_at=0.0, dropped_frames=0)
        result, overlay = session._process_packet(packet)

        self.assertEqual(result["type"], "pose_result")
        self.assertTrue(result["has_overlay"])
        self.assertEqual(result["current_pose"]["keypoint_profile"], "coco17")
        self.assertIsNotNone(overlay)
        self.assertGreater(len(overlay), 0)


if __name__ == "__main__":
    unittest.main()
