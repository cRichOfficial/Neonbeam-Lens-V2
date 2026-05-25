from __future__ import annotations

import io
import math

import cv2
import numpy as np
from fastapi import HTTPException
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from app.config import get_config_store
from app.schemas.calibration import AprilTagPdfRequest

LETTER_WIDTH_MM = 215.9
LETTER_HEIGHT_MM = 279.4
TAG_IDS = [0, 1, 2, 3]
LABEL_HEIGHT_MM = 6.0
GUTTER_MM = 8.0


def _mm_to_pt(value_mm: float) -> float:
    return value_mm * mm


def _generate_tag_image(tag_id: int, pixel_size: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    image = cv2.aruco.generateImageMarker(dictionary, tag_id, pixel_size, borderBits=1)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def _validate_layout(size_mm: float, safe_zone_padding_mm: float, page_margin_mm: float) -> None:
    unit_mm = size_mm + (2 * safe_zone_padding_mm)
    label_unit_mm = unit_mm + LABEL_HEIGHT_MM
    required_width = (2 * unit_mm) + GUTTER_MM
    required_height = (2 * label_unit_mm) + GUTTER_MM
    printable_width = LETTER_WIDTH_MM - (2 * page_margin_mm)
    printable_height = LETTER_HEIGHT_MM - (2 * page_margin_mm)

    if required_width > printable_width or required_height > printable_height:
        max_by_width = (
            (printable_width - GUTTER_MM) / 2 - (2 * safe_zone_padding_mm) - (LABEL_HEIGHT_MM / 2)
        )
        max_by_height = (
            (printable_height - GUTTER_MM) / 2 - (2 * safe_zone_padding_mm) - LABEL_HEIGHT_MM
        )
        max_size = max(0.0, min(max_by_width, max_by_height))
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Tag layout does not fit on Letter page at requested size",
                "max_size_mm": round(max_size, 2),
                "requested_size_mm": size_mm,
                "safe_zone_padding_mm": safe_zone_padding_mm,
            },
        )


def generate_apriltag_pdf(request: AprilTagPdfRequest) -> bytes:
    config = get_config_store().config.apriltag
    page_margin_mm = config.pdf_page_margin_mm
    _validate_layout(request.size_mm, request.safe_zone_padding_mm, page_margin_mm)

    unit_mm = request.size_mm + (2 * request.safe_zone_padding_mm)
    label_unit_mm = unit_mm + LABEL_HEIGHT_MM
    grid_width = (2 * unit_mm) + GUTTER_MM
    grid_height = (2 * label_unit_mm) + GUTTER_MM
    origin_x_mm = (LETTER_WIDTH_MM - grid_width) / 2.0
    origin_y_mm = (LETTER_HEIGHT_MM - grid_height) / 2.0

    positions = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    ]

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    page_width_pt, page_height_pt = letter

    pdf.setTitle("AprilTags 0-3")
    pdf.setAuthor("Laser Object Detection")

    pixel_size = max(200, int(math.ceil(request.size_mm * 20)))

    for index, (col, row) in enumerate(positions):
        tag_id = TAG_IDS[index]
        x_mm = origin_x_mm + col * (unit_mm + GUTTER_MM)
        y_mm = origin_y_mm + row * (label_unit_mm + GUTTER_MM)

        x_pt = _mm_to_pt(x_mm)
        y_pt = page_height_pt - _mm_to_pt(y_mm + unit_mm)
        unit_w_pt = _mm_to_pt(unit_mm)
        unit_h_pt = _mm_to_pt(unit_mm)
        tag_pt = _mm_to_pt(request.size_mm)
        pad_pt = _mm_to_pt(request.safe_zone_padding_mm)

        pdf.setDash(3, 2)
        pdf.setLineWidth(0.6)
        pdf.rect(x_pt, y_pt, unit_w_pt, unit_h_pt, stroke=1, fill=0)
        pdf.setDash()

        tag_x_pt = x_pt + pad_pt
        tag_y_pt = y_pt + pad_pt
        tag_image = _generate_tag_image(tag_id, pixel_size)
        pdf.drawImage(
            ImageReader(_ndarray_to_png_bytes(tag_image)),
            tag_x_pt,
            tag_y_pt,
            width=tag_pt,
            height=tag_pt,
            preserveAspectRatio=True,
            anchor="sw",
        )

        label_y_pt = y_pt - _mm_to_pt(4.0)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(x_pt, label_y_pt, f"ID: {tag_id}")

    scale_bar_mm = 10.0
    bar_x_mm = origin_x_mm
    bar_y_mm = page_margin_mm / 2.0
    bar_x_pt = _mm_to_pt(bar_x_mm)
    bar_y_pt = _mm_to_pt(bar_y_mm)
    bar_w_pt = _mm_to_pt(scale_bar_mm)
    pdf.setLineWidth(1.2)
    pdf.line(bar_x_pt, bar_y_pt, bar_x_pt + bar_w_pt, bar_y_pt)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(bar_x_pt, bar_y_pt + 4, f"{int(scale_bar_mm)} mm scale bar (print at 100%)")

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def _ndarray_to_png_bytes(image: np.ndarray) -> io.BytesIO:
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode tag image")
    out = io.BytesIO(encoded.tobytes())
    out.seek(0)
    return out
