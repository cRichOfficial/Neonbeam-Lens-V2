from __future__ import annotations

import json
from datetime import datetime, timezone

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import ConfigStore, PROJECT_ROOT
from app.main import create_app
from app.schemas.common import BoundingBox, Point2D
from app.schemas.dataset import AnnotationShape
from app.services import dataset_service as dataset_service_module
from app.services.dataset_service import DatasetService


@pytest.fixture
def dataset_env(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    dataset_root = tmp_path / "dataset"
    config_path.write_text(
        f"""
bed:
  width_mm: 400
  height_mm: 400
camera:
  mount_height_mm: 360
  exposure_us: 10000
  analogue_gain: 1.0
  main_resolution: [640, 480]
  lores_resolution: [640, 640]
dataset:
  storage_path: {dataset_root.as_posix()}
  train_val_split: 0.5
  classes: ['part_a', 'part_b']
""".strip(),
        encoding="utf-8",
    )
    store = ConfigStore(settings=type("S", (), {"config_path": config_path})())
    monkeypatch.setattr("app.config.get_config_store", lambda: store)
    monkeypatch.setattr("app.config._config_store", store)
    monkeypatch.setattr("app.services.dataset_service.get_config_store", lambda: store)
    dataset_service_module._dataset_service = None
    yield dataset_root
    dataset_service_module._dataset_service = None


@pytest.fixture
def client(dataset_env):
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _write_test_image(service: DatasetService) -> str:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[:] = (40, 80, 120)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    image_id = "test-image-1"
    filename = f"{image_id}.jpg"
    (service.images_dir / filename).write_bytes(encoded.tobytes())
    service._save_thumbnail(image_id, image)
    service._save_preview(image_id, image)
    record = {
        "id": image_id,
        "filename": filename,
        "width": 120,
        "height": 100,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "reviewed": True,
        "notes": "",
        "annotations": [],
    }
    (service.metadata_dir / f"{image_id}.json").write_text(json.dumps(record), encoding="utf-8")
    return image_id


def test_get_and_update_classes(client: TestClient) -> None:
    response = client.get("/api/v1/dataset/classes")
    assert response.status_code == 200
    assert response.json()["classes"] == ["part_a", "part_b"]

    response = client.put("/api/v1/dataset/classes", json={"classes": ["wood_blank", "jig"]})
    assert response.status_code == 200
    assert response.json()["classes"] == ["wood_blank", "jig"]


def test_save_annotations(client: TestClient, dataset_env) -> None:
    service = DatasetService()
    image_id = _write_test_image(service)
    payload = {
        "annotations": [
            {
                "id": "ann-1",
                "class_id": 0,
                "type": "bbox",
                "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.7},
                "polygon": [],
            }
        ]
    }
    response = client.put(f"/api/v1/dataset/images/{image_id}/annotations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["annotations"]) == 1
    assert data["annotations"][0]["type"] == "bbox"


def test_bbox_to_polygon_conversion() -> None:
    bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.7)
    polygon = DatasetService.bbox_to_polygon(bbox)
    assert len(polygon) == 4
    assert polygon[0] == Point2D(x=0.1, y=0.2)
    line = DatasetService.bbox_to_yolo_line(1, bbox)
    assert line.startswith("1 ")


def test_export_dual_datasets(client: TestClient, dataset_env) -> None:
    service = DatasetService()
    image_id = _write_test_image(service)
    ann = AnnotationShape(
        id="ann-1",
        class_id=0,
        type="bbox",
        bbox=BoundingBox(x_min=0.1, y_min=0.2, x_max=0.4, y_max=0.6),
    )
    service.save_annotations(image_id, [ann])

    ann2 = AnnotationShape(
        id="ann-2",
        class_id=1,
        type="polygon",
        bbox=BoundingBox(x_min=0.2, y_min=0.3, x_max=0.8, y_max=0.9),
        polygon=[
            Point2D(x=0.2, y=0.3),
            Point2D(x=0.8, y=0.35),
            Point2D(x=0.7, y=0.9),
        ],
    )
    service.save_annotations(image_id, [ann, ann2])
    service.patch_image(image_id, reviewed=True)

    response = client.post("/api/v1/dataset/export", json={"reviewed_only": True, "seed": 42})
    assert response.status_code == 200
    data = response.json()
    assert data["detection"]["train_images"] >= 1
    assert data["segmentation"]["val_images"] >= 0

    det_label = dataset_env / "export" / "detection" / "labels" / "train" / "test-image-1.txt"
    seg_label = dataset_env / "export" / "segmentation" / "labels" / "train" / "test-image-1.txt"
    assert det_label.exists()
    assert seg_label.exists()
    det_lines = det_label.read_text(encoding="utf-8").strip().splitlines()
    seg_lines = seg_label.read_text(encoding="utf-8").strip().splitlines()
    assert len(det_lines) == 2
    assert len(seg_lines) == 2
    assert len(seg_lines[1].split()) > 5


def test_capture_requires_classes(client: TestClient, dataset_env) -> None:
    response = client.put("/api/v1/dataset/classes", json={"classes": ["only"]})
    assert response.status_code == 200
    response = client.post("/api/v1/dataset/capture")
    assert response.status_code == 200
    assert "id" in response.json()


def test_image_variants(client: TestClient, dataset_env) -> None:
    service = DatasetService()
    image_id = _write_test_image(service)

    full = client.get(f"/api/v1/dataset/images/{image_id}/file?variant=full")
    thumb = client.get(f"/api/v1/dataset/images/{image_id}/file?variant=thumb")
    preview = client.get(f"/api/v1/dataset/images/{image_id}/file?variant=preview")

    assert full.status_code == 200
    assert thumb.status_code == 200
    assert preview.status_code == 200

    full_arr = cv2.imdecode(np.frombuffer(full.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    thumb_arr = cv2.imdecode(np.frombuffer(thumb.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert full_arr.shape[0] == 100
    assert max(thumb_arr.shape[:2]) <= 256


def test_preview_cached_on_disk(client: TestClient, dataset_env) -> None:
    service = DatasetService()
    image_id = _write_test_image(service)
    preview_path = dataset_env / "images" / "previews" / f"{image_id}.jpg"
    assert preview_path.exists()

    first = client.get(f"/api/v1/dataset/images/{image_id}/file?variant=preview")
    assert first.status_code == 200
    assert first.content == preview_path.read_bytes()


def test_dataset_stats(client: TestClient, dataset_env) -> None:
    service = DatasetService()
    image_id = _write_test_image(service)
    ann = AnnotationShape(
        id="ann-1",
        class_id=0,
        type="bbox",
        bbox=BoundingBox(x_min=0.1, y_min=0.2, x_max=0.4, y_max=0.6),
    )
    service.save_annotations(image_id, [ann])
    service.patch_image(image_id, reviewed=True)

    response = client.get("/api/v1/dataset/stats?reviewed_only=true")
    assert response.status_code == 200
    data = response.json()
    assert data["reviewed_images"] == 1
    assert data["class_counts"]["part_a"] == 1
    assert data["train_val_split"] == 0.5
