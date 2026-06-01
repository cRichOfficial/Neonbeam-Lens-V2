"""JPEG encoding from in-memory RGB frames."""

from __future__ import annotations

import cv2
import numpy as np

from app.services.image_encoding import encode_jpeg_bgr, encode_jpeg_rgb
from app.services.pipeline_debug_mosaic import compose_stage_mosaic, encode_jpeg
from app.services.work_area_renderer import WorkAreaRenderer, WorkAreaView


def _decoded_center_channel(jpeg: bytes) -> tuple[int, int, int]:
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    b, g, r = decoded[decoded.shape[0] // 2, decoded.shape[1] // 2]
    return int(b), int(g), int(r)


def _solid_red_rgb(shape: tuple[int, int] = (32, 32)) -> np.ndarray:
    frame = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    frame[:, :] = (255, 0, 0)
    return frame


def test_encode_jpeg_rgb_preserves_red_channel() -> None:
    jpeg = encode_jpeg_rgb(_solid_red_rgb(), quality=90)
    b, _g, r = _decoded_center_channel(jpeg)
    assert r > b
    assert r > 200


def test_work_area_encode_jpeg_preserves_red_channel() -> None:
    image = _solid_red_rgb((64, 80))
    view = WorkAreaView(
        image=image,
        width_mm=100.0,
        height_mm=80.0,
        width_px=image.shape[1],
        height_px=image.shape[0],
        pixels_per_mm=0.8,
        origin_tag_id=0,
    )
    jpeg = WorkAreaRenderer().encode_jpeg(view, quality=90)
    b, _g, r = _decoded_center_channel(jpeg)
    assert r > b
    assert r > 200


def test_pipeline_debug_encode_preserves_red_channel() -> None:
    jpeg = encode_jpeg(_solid_red_rgb(), quality=90)
    b, _g, r = _decoded_center_channel(jpeg)
    assert r > b
    assert r > 200


def test_mosaic_compose_red_tile_preserves_red_after_encode() -> None:
    red = _solid_red_rgb((120, 160))
    mosaic = compose_stage_mosaic([("red", red)], max_width_px=320, max_height_px=240, columns=1)
    jpeg = encode_jpeg_bgr(mosaic, quality=90)
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    b, _g, r = cv2.split(decoded)
    assert int(r.max()) > int(b.max())
    assert int(r.max()) > 200
