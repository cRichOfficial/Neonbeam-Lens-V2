from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from app.schemas.dataset import (
    AnnotationsUpdateRequest,
    ClassesResponse,
    ClassesUpdateRequest,
    DatasetStatsResponse,
    ExportRequest,
    ExportResponse,
    ExportStatusResponse,
    ImageListResponse,
    ImagePatchRequest,
    ImageRecord,
)
from app.services.dataset_service import DatasetService, get_dataset_service

router = APIRouter(prefix="/api/v1/dataset", tags=["dataset"])


@router.get("/classes", response_model=ClassesResponse)
def get_classes(service: DatasetService = Depends(get_dataset_service)) -> ClassesResponse:
    return ClassesResponse(classes=service.get_classes())


@router.put("/classes", response_model=ClassesResponse)
def update_classes(
    payload: ClassesUpdateRequest,
    service: DatasetService = Depends(get_dataset_service),
) -> ClassesResponse:
    try:
        classes = service.set_classes(payload.classes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ClassesResponse(classes=classes)


@router.post("/capture", response_model=ImageRecord)
def capture_image(service: DatasetService = Depends(get_dataset_service)) -> ImageRecord:
    try:
        return service.capture_from_camera()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/images", response_model=ImageListResponse)
def list_images(
    reviewed: bool | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: DatasetService = Depends(get_dataset_service),
) -> ImageListResponse:
    return service.list_images(reviewed=reviewed, limit=limit, offset=offset)


@router.get("/images/{image_id}", response_model=ImageRecord)
def get_image(
    image_id: str,
    service: DatasetService = Depends(get_dataset_service),
) -> ImageRecord:
    try:
        return service.get_image(image_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/stats", response_model=DatasetStatsResponse)
def dataset_stats(
    reviewed_only: bool = Query(default=True),
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetStatsResponse:
    return service.get_stats(reviewed_only=reviewed_only)


@router.get("/images/{image_id}/file")
def get_image_file(
    image_id: str,
    variant: str = Query(default="full", pattern="^(full|preview|thumb)$"),
    service: DatasetService = Depends(get_dataset_service),
) -> Response:
    try:
        content = service.get_image_bytes(image_id, variant=variant)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=content, media_type="image/jpeg")


@router.put("/images/{image_id}/annotations", response_model=ImageRecord)
def update_annotations(
    image_id: str,
    payload: AnnotationsUpdateRequest,
    service: DatasetService = Depends(get_dataset_service),
) -> ImageRecord:
    try:
        return service.save_annotations(image_id, payload.annotations)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/images/{image_id}", response_model=ImageRecord)
def patch_image(
    image_id: str,
    payload: ImagePatchRequest,
    service: DatasetService = Depends(get_dataset_service),
) -> ImageRecord:
    try:
        return service.patch_image(image_id, reviewed=payload.reviewed, notes=payload.notes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/images/{image_id}", status_code=204)
def delete_image(
    image_id: str,
    service: DatasetService = Depends(get_dataset_service),
) -> Response:
    try:
        service.delete_image(image_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


@router.post("/export", response_model=ExportResponse)
def export_datasets(
    payload: ExportRequest = ExportRequest(),
    service: DatasetService = Depends(get_dataset_service),
) -> ExportResponse:
    try:
        return service.export_datasets(reviewed_only=payload.reviewed_only, seed=payload.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/export/status", response_model=ExportStatusResponse)
def export_status(service: DatasetService = Depends(get_dataset_service)) -> ExportStatusResponse:
    last = service.get_export_status()
    return ExportStatusResponse(
        last_export_at=last.exported_at if last else None,
        last_export=last,
    )
