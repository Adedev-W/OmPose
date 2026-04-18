from __future__ import annotations

import base64
from pathlib import Path

from src.errors import ImageInputError


SUPPORTED_IMAGE_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def detect_image_mime(image_path: Path) -> str:
    suffix_mime = SUPPORTED_IMAGE_MIME_BY_SUFFIX.get(image_path.suffix.lower())
    header = image_path.read_bytes()[:16]

    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    if suffix_mime is not None:
        return suffix_mime
    raise ImageInputError(f"unsupported image format: {image_path}")


def image_to_data_uri(image_path: Path) -> str:
    mime = detect_image_mime(image_path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def load_image_bgr(image_path: Path):
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImageInputError("opencv-python-headless is required to load images") from exc

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageInputError(f"failed to read image: {image_path}")
    return image

