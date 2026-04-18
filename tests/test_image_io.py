from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from src.image_io import detect_image_mime, image_to_data_uri


class ImageIOTests(unittest.TestCase):
    def test_detects_supported_mime_types(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jpg = root / "sample.jpg"
            png = root / "sample.png"
            webp = root / "sample.webp"
            jpg.write_bytes(b"\xff\xd8\xff\xe0demo")
            png.write_bytes(b"\x89PNG\r\n\x1a\nrest")
            webp.write_bytes(b"RIFF1234WEBPrest")

            self.assertEqual(detect_image_mime(jpg), "image/jpeg")
            self.assertEqual(detect_image_mime(png), "image/png")
            self.assertEqual(detect_image_mime(webp), "image/webp")

    def test_image_to_data_uri(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.jpg"
            payload = b"\xff\xd8\xff\xe0demo"
            image_path.write_bytes(payload)
            uri = image_to_data_uri(image_path)

            self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
            encoded = uri.split(",", 1)[1]
            self.assertEqual(base64.b64decode(encoded), payload)


if __name__ == "__main__":
    unittest.main()

