from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.marker_renderer import MarkerRenderer
from src.pipeline import OmPosePipeline
from src.streaming.session import StreamingPoseSession
from src.yolo_pose_detector import YoloPoseDetector, YoloPoseSettings


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_CORS_ORIGINS = "http://127.0.0.1:5173,http://localhost:5173,http://0.0.0.0:5173"
DEFAULT_CORS_ORIGIN_REGEX = (
    r"^https?://("
    r"localhost|127\.0\.0\.1|0\.0\.0\.0|"
    r"10\.\d+\.\d+\.\d+|"
    r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|"
    r"192\.168\.\d+\.\d+"
    r"):\d+$"
)


def cors_origin_regex_from_env() -> str:
    value = os.getenv("OMPOSE_CORS_ORIGIN_REGEX", DEFAULT_CORS_ORIGIN_REGEX)
    return value.replace("\\\\", "\\")


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_path: str
    out_dir: str = "output"
    recommendation_count: int = Field(default=3, ge=1, le=10)
    pose_index: int = Field(default=0, ge=0)


def validate_upload_request(file: UploadFile, recommendation_count: int, pose_index: int) -> str:
    if recommendation_count < 1 or recommendation_count > 10:
        raise HTTPException(status_code=422, detail="recommendation_count must be between 1 and 10")
    if pose_index < 0:
        raise HTTPException(status_code=422, detail="pose_index must be non-negative")
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail=f"unsupported image content type: {file.content_type}")
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }[file.content_type]


def create_app(detector: Any | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.renderer = MarkerRenderer()
        app.state.stream_semaphore = asyncio.Semaphore(int(os.getenv("OMPOSE_MAX_STREAMS_PER_PROCESS", "1")))
        app.state.recommendation_jobs = {}
        app.state.pose_guide_jobs = {}
        app.state.recommendation_jobs_lock = asyncio.Lock()
        app.state.captures: list[dict[str, Any]] = []
        app.state.captures_lock = asyncio.Lock()
        if detector is None:
            app.state.detector = YoloPoseDetector(YoloPoseSettings.from_env())
            await asyncio.to_thread(app.state.detector.load)
        else:
            app.state.detector = detector
            if hasattr(app.state.detector, "load"):
                await asyncio.to_thread(app.state.detector.load)
        yield

    app = FastAPI(title="OmPose Streaming API", version="0.2.0", lifespan=lifespan)
    cors_origins = [
        origin.strip()
        for origin in os.getenv("OMPOSE_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=cors_origin_regex_from_env(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        active_detector = app.state.detector
        metadata = active_detector.metadata() if hasattr(active_detector, "metadata") else {}
        return {
            "status": "ok",
            "service": "ompose-streaming",
            "detector_backend": metadata.get("backend", "unknown"),
            "model_loaded": metadata.get("model_loaded", False),
            "model_path": metadata.get("model_path"),
        }

    @app.get("/models/current")
    async def current_model() -> dict[str, Any]:
        active_detector = app.state.detector
        if hasattr(active_detector, "metadata"):
            return active_detector.metadata()
        return {"backend": "unknown", "model_loaded": False}

    def run_pipeline_for_path(image_path: Path, out_dir: Path, pose_index: int, recommendation_count: int):
        pipeline = OmPosePipeline(pose_detector=app.state.detector)
        return pipeline.run(
            image_path=image_path,
            out_dir=out_dir,
            pose_index=pose_index,
            recommendation_count=recommendation_count,
        )

    def run_pose_guide_for_path(image_path: Path, pose_index: int):
        pipeline = OmPosePipeline(pose_detector=app.state.detector)
        return pipeline.run_pose_guide(image_path=image_path, pose_index=pose_index)

    async def save_upload_to_temp(file: UploadFile, suffix: str) -> Path:
        with tempfile.NamedTemporaryFile(prefix="ompose-upload-", suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            while chunk := await file.read(1024 * 1024):
                temp_file.write(chunk)
        return temp_path

    async def update_recommendation_job(job_id: str, **values: Any) -> None:
        async with app.state.recommendation_jobs_lock:
            job = app.state.recommendation_jobs[job_id]
            job.update(values)

    async def run_recommendation_job(
        job_id: str,
        temp_path: Path,
        recommendation_count: int,
        pose_index: int,
    ) -> None:
        await update_recommendation_job(job_id, status="running")
        try:
            result = await asyncio.to_thread(
                run_pipeline_for_path,
                temp_path,
                Path("output"),
                pose_index,
                recommendation_count,
            )
            await update_recommendation_job(job_id, status="completed", result=result.model_dump(), error=None)
        except Exception as exc:
            LOGGER.exception("async upload recommendation pipeline failed")
            await update_recommendation_job(job_id, status="failed", error=str(exc), result=None)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    async def run_pose_guide_job(job_id: str, temp_path: Path, pose_index: int) -> None:
        await update_job(app.state.pose_guide_jobs, job_id, status="running")
        try:
            result = await asyncio.to_thread(run_pose_guide_for_path, temp_path, pose_index)
            await update_job(app.state.pose_guide_jobs, job_id, status="completed", result=result.model_dump(), error=None)
        except Exception as exc:
            LOGGER.exception("async pose guide pipeline failed")
            await update_job(app.state.pose_guide_jobs, job_id, status="failed", error=str(exc), result=None)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    async def update_job(job_store: dict[str, Any], job_id: str, **values: Any) -> None:
        async with app.state.recommendation_jobs_lock:
            job = job_store[job_id]
            job.update(values)

    @app.post("/api/recommendations")
    async def create_recommendations(request: RecommendationRequest) -> dict[str, Any]:
        image_path = Path(request.image_path)
        if not image_path.exists() or not image_path.is_file():
            raise HTTPException(status_code=404, detail=f"image does not exist: {image_path}")

        try:
            result = await asyncio.to_thread(
                run_pipeline_for_path,
                image_path,
                Path(request.out_dir),
                request.pose_index,
                request.recommendation_count,
            )
        except Exception as exc:
            LOGGER.exception("recommendation pipeline failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return result.model_dump()

    @app.post("/api/recommendations/upload")
    async def create_recommendations_from_upload(
        file: UploadFile = File(...),
        recommendation_count: int = Form(3),
        pose_index: int = Form(0),
    ) -> dict[str, Any]:
        suffix = validate_upload_request(file, recommendation_count, pose_index)
        temp_path: Path | None = None
        try:
            temp_path = await save_upload_to_temp(file, suffix)
            result = await asyncio.to_thread(
                run_pipeline_for_path,
                temp_path,
                Path("output"),
                pose_index,
                recommendation_count,
            )
            return result.model_dump()
        except Exception as exc:
            LOGGER.exception("upload recommendation pipeline failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            await file.close()
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    @app.post("/api/recommendations/upload/async")
    async def start_recommendations_from_upload(
        file: UploadFile = File(...),
        recommendation_count: int = Form(3),
        pose_index: int = Form(0),
    ) -> dict[str, Any]:
        suffix = validate_upload_request(file, recommendation_count, pose_index)
        temp_path: Path | None = None
        try:
            temp_path = await save_upload_to_temp(file, suffix)
            job_id = uuid.uuid4().hex
            async with app.state.recommendation_jobs_lock:
                app.state.recommendation_jobs[job_id] = {
                    "job_id": job_id,
                    "status": "queued",
                    "result": None,
                    "error": None,
                }
            asyncio.create_task(run_recommendation_job(job_id, temp_path, recommendation_count, pose_index))
            temp_path = None
            return {"job_id": job_id, "status": "queued"}
        except Exception as exc:
            LOGGER.exception("failed to queue upload recommendation job")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            await file.close()
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    @app.get("/api/recommendations/jobs/{job_id}")
    async def get_recommendation_job(job_id: str) -> dict[str, Any]:
        async with app.state.recommendation_jobs_lock:
            job = app.state.recommendation_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"recommendation job does not exist: {job_id}")
            return dict(job)

    @app.post("/api/pose-guide/upload/async")
    async def start_pose_guide_from_upload(
        file: UploadFile = File(...),
        pose_index: int = Form(0),
    ) -> dict[str, Any]:
        suffix = validate_upload_request(file, recommendation_count=1, pose_index=pose_index)
        temp_path: Path | None = None
        try:
            temp_path = await save_upload_to_temp(file, suffix)
            job_id = uuid.uuid4().hex
            async with app.state.recommendation_jobs_lock:
                app.state.pose_guide_jobs[job_id] = {
                    "job_id": job_id,
                    "status": "queued",
                    "result": None,
                    "error": None,
                }
            asyncio.create_task(run_pose_guide_job(job_id, temp_path, pose_index))
            temp_path = None
            return {"job_id": job_id, "status": "queued"}
        except Exception as exc:
            LOGGER.exception("failed to queue pose guide job")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            await file.close()
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    @app.get("/api/pose-guide/jobs/{job_id}")
    async def get_pose_guide_job(job_id: str) -> dict[str, Any]:
        async with app.state.recommendation_jobs_lock:
            job = app.state.pose_guide_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"pose guide job does not exist: {job_id}")
            return dict(job)

    # ── Capture Photo (in-memory, no database) ──────────────────────

    @app.post("/api/capture")
    async def capture_photo(
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        """Capture a photo from the camera. Stores in-memory only — not persisted to disk or database."""
        if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(status_code=415, detail=f"unsupported image content type: {file.content_type}")

        image_bytes = await file.read()
        await file.close()

        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="empty image upload")

        capture_id = uuid.uuid4().hex[:12]
        b64_data = base64.b64encode(image_bytes).decode("ascii")
        mime = file.content_type or "image/jpeg"
        captured_at = datetime.now(timezone.utc).isoformat()

        capture_record: dict[str, Any] = {
            "capture_id": capture_id,
            "data_url": f"data:{mime};base64,{b64_data}",
            "mime_type": mime,
            "size_bytes": len(image_bytes),
            "captured_at": captured_at,
        }

        async with app.state.captures_lock:
            app.state.captures.append(capture_record)
            # Keep at most 50 captures in memory to avoid OOM
            if len(app.state.captures) > 50:
                app.state.captures = app.state.captures[-50:]

        LOGGER.info("captured photo %s (%d bytes)", capture_id, len(image_bytes))
        return capture_record

    @app.get("/api/captures")
    async def list_captures() -> dict[str, Any]:
        """Return all in-memory captured photos."""
        async with app.state.captures_lock:
            return {"captures": list(app.state.captures), "count": len(app.state.captures)}

    @app.delete("/api/captures/{capture_id}")
    async def delete_capture(capture_id: str) -> dict[str, Any]:
        """Delete a single capture from memory."""
        async with app.state.captures_lock:
            before = len(app.state.captures)
            app.state.captures = [c for c in app.state.captures if c["capture_id"] != capture_id]
            if len(app.state.captures) == before:
                raise HTTPException(status_code=404, detail=f"capture not found: {capture_id}")
        return {"deleted": capture_id}

    @app.delete("/api/captures")
    async def clear_captures() -> dict[str, Any]:
        """Clear all captures from memory."""
        async with app.state.captures_lock:
            count = len(app.state.captures)
            app.state.captures.clear()
        return {"cleared": count}

    @app.websocket("/ws/pose")
    async def pose_websocket(websocket: WebSocket) -> None:
        semaphore = app.state.stream_semaphore
        if semaphore.locked():
            await websocket.accept()
            await websocket.send_json({"type": "error", "message": "stream capacity is full"})
            await websocket.close(code=1013)
            return

        async with semaphore:
            session = StreamingPoseSession(
                websocket=websocket,
                detector=app.state.detector,
                renderer=app.state.renderer,
                default_return_overlay=os.getenv("OMPOSE_STREAM_RETURN_OVERLAY", "true").lower() == "true",
                default_max_fps=float(os.getenv("OMPOSE_STREAM_MAX_FPS", "10")),
            )
            await session.run()

    return app


app = create_app()
