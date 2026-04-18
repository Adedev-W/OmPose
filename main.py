from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is installed in normal runtime.
    load_dotenv = None

from src.errors import ConfigError, ImageInputError, OmPoseError
from src.pipeline import OmPosePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ompose",
        description="OmPose MVP v1 CLI: image -> pose detection -> VLM pose recommendation -> marker overlay.",
    )
    parser.add_argument("image_path", help="Path to a JPEG, PNG, or WEBP image.")
    parser.add_argument("--out", default="output", help="Output directory for overlay image and result JSON.")
    parser.add_argument("--model", default=None, help="DashScope/OpenAI-compatible model override.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL override.")
    parser.add_argument("--pose-index", type=int, default=0, help="Detected pose index to use. MVP defaults to 0.")
    parser.add_argument(
        "--model-task-path",
        default=None,
        help="Path to MediaPipe pose_landmarker .task asset. Defaults to assets/models/pose_landmarker_heavy.task.",
    )
    parser.add_argument("--recommendations", type=int, default=5, help="Number of pose recommendations to request.")
    return parser


def main(argv: list[str] | None = None) -> int:
    if load_dotenv is not None:
        load_dotenv()

    parser = build_parser()
    args = parser.parse_args(argv)

    image_path = Path(args.image_path)
    if not image_path.exists() or not image_path.is_file():
        print(f"Image input error: file does not exist: {image_path}", file=sys.stderr)
        return ImageInputError.exit_code

    if args.recommendations < 1:
        print("Config error: --recommendations must be at least 1", file=sys.stderr)
        return ConfigError.exit_code

    try:
        pipeline = OmPosePipeline(
            model=args.model,
            base_url=args.base_url,
            model_task_path=Path(args.model_task_path) if args.model_task_path else None,
        )
        result = pipeline.run(
            image_path=image_path,
            out_dir=Path(args.out),
            pose_index=args.pose_index,
            recommendation_count=args.recommendations,
        )
    except OmPoseError as exc:
        print(f"{exc.label}: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - last-resort CLI guard.
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 99

    print(f"Overlay image: {result.overlay_image}")
    print(f"Contact sheet: {result.contact_sheet_image}")
    print(f"Result JSON: {result.result_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
