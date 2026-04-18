from __future__ import annotations

from pathlib import Path

from src.errors import OverlaySaveError
from src.image_io import load_image_bgr
from src.marker_renderer import MarkerRenderer
from src.models.schemas import PipelineResult, PoseGuideResult
from src.pose_detector import PoseDetector
from src.pose_guide_engine import PoseGuideEngine
from src.vlm_reasoner import VLMReasoner


class OmPosePipeline:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        model_task_path: Path | None = None,
        pose_detector: PoseDetector | None = None,
        vlm_reasoner: VLMReasoner | None = None,
        marker_renderer: MarkerRenderer | None = None,
        pose_guide_engine: PoseGuideEngine | None = None,
    ) -> None:
        self.pose_detector = pose_detector or PoseDetector(model_task_path=model_task_path)
        if vlm_reasoner is None:
            self.vlm_reasoner = VLMReasoner(model=model, base_url=base_url)
            self.vlm_reasoner.validate_config()
        else:
            self.vlm_reasoner = vlm_reasoner
        self.marker_renderer = marker_renderer or MarkerRenderer()
        self.pose_guide_engine = pose_guide_engine or PoseGuideEngine()

    def run(
        self,
        image_path: Path,
        out_dir: Path,
        pose_index: int = 0,
        recommendation_count: int = 3,
    ) -> PipelineResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        image_bgr = load_image_bgr(image_path)
        current_pose = self.pose_detector.detect(image_path, pose_index=pose_index)
        if hasattr(self.vlm_reasoner, "select_pose_guide"):
            guide_result, usage = self.vlm_reasoner.select_pose_guide(image_path=image_path, current_pose=current_pose)
            recommendations = [self.pose_guide_engine.generate(current_pose, guide_result.recommendation)]
            scene = guide_result.scene
        else:
            reasoning_result, usage = self.vlm_reasoner.recommend(
                image_path=image_path,
                current_pose=current_pose,
                recommendation_count=recommendation_count,
            )
            recommendations = reasoning_result.recommendations
            scene = reasoning_result.scene
        selected_index = 0
        selected = recommendations[selected_index]
        overlay = self.marker_renderer.render(image_bgr, current_pose, selected)
        contact_sheet = self.marker_renderer.render_contact_sheet(
            image_bgr,
            current_pose,
            recommendations,
        )

        overlay_path = out_dir / f"{image_path.stem}_overlay.jpg"
        contact_sheet_path = out_dir / f"{image_path.stem}_contact_sheet.jpg"
        result_path = out_dir / f"{image_path.stem}_result.json"
        self._save_overlay(overlay_path, overlay)
        self._save_overlay(contact_sheet_path, contact_sheet)

        result = PipelineResult(
            input_image=str(image_path),
            overlay_image=str(overlay_path),
            contact_sheet_image=str(contact_sheet_path),
            result_json=str(result_path),
            scene=scene,
            current_pose=current_pose,
            recommendations=recommendations,
            selected_recommendation_index=selected_index,
            usage=usage,
        )
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    def run_pose_guide(
        self,
        image_path: Path,
        pose_index: int = 0,
    ) -> PoseGuideResult:
        image_bgr = load_image_bgr(image_path)
        current_pose = self.pose_detector.detect(image_path, pose_index=pose_index)
        guide_result, usage = self.vlm_reasoner.select_pose_guide(image_path=image_path, current_pose=current_pose)
        target_pose = self.pose_guide_engine.generate(current_pose, guide_result.recommendation)
        return PoseGuideResult(
            input_image=str(image_path),
            scene=guide_result.scene,
            current_pose=current_pose,
            selected_pose_guide=guide_result.recommendation,
            target_pose=target_pose,
            usage=usage,
        )

    def _save_overlay(self, overlay_path: Path, overlay) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise OverlaySaveError("opencv-python-headless is required to save overlay images") from exc

        ok = cv2.imwrite(str(overlay_path), overlay)
        if not ok:
            raise OverlaySaveError(f"failed to save overlay image: {overlay_path}")
