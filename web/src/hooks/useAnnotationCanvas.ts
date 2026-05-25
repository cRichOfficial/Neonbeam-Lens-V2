import { useCallback, useEffect, useRef, useState } from "react";
import type { AnnotationShape, ActiveTool, BoundingBox, Point2D } from "../api/types";
import { bboxFromPoints, bboxFromPolygon, classColor, clamp01, uuid } from "../utils/geometry";

export interface CanvasLayout {
  offsetX: number;
  offsetY: number;
  drawWidth: number;
  drawHeight: number;
}

export interface ViewTransform {
  scale: number;
  panX: number;
  panY: number;
}

export interface LetterboxStyle {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface UseAnnotationCanvasOptions {
  imageUrl: string | null;
  annotations: AnnotationShape[];
  selectedId: string | null;
  activeClassId: number;
  activeTool: ActiveTool;
  onCommit: (annotations: AnnotationShape[]) => void;
  onSelect: (id: string | null) => void;
  onBeforeChange: () => void;
}

type DragMode = "create-bbox" | "move" | "vertex" | "pan-view" | null;

const MIN_ZOOM = 1;
const MAX_ZOOM = 5;
const TOUCH_HIT_PX = 24;
const MOUSE_HIT_PX = 8;

function clampZoom(scale: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, scale));
}

export function useAnnotationCanvas({
  imageUrl,
  annotations,
  selectedId,
  activeClassId,
  activeTool,
  onCommit,
  onSelect,
  onBeforeChange,
}: UseAnnotationCanvasOptions) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const layoutRef = useRef<CanvasLayout>({ offsetX: 0, offsetY: 0, drawWidth: 0, drawHeight: 0 });
  const viewRef = useRef<ViewTransform>({ scale: 1, panX: 0, panY: 0 });
  const letterboxStyleRef = useRef<LetterboxStyle>({ left: 0, top: 0, width: 0, height: 0 });
  const [letterboxStyle, setLetterboxStyle] = useState<LetterboxStyle>({
    left: 0,
    top: 0,
    width: 0,
    height: 0,
  });
  const [viewTransform, setViewTransform] = useState<ViewTransform>({ scale: 1, panX: 0, panY: 0 });

  const dragModeRef = useRef<DragMode>(null);
  const dragStartRef = useRef<Record<string, unknown> | null>(null);
  const tempBboxRef = useRef<BoundingBox | null>(null);
  const tempPolygonRef = useRef<Point2D[]>([]);
  const annotationsRef = useRef(annotations);
  const rafRef = useRef<number | null>(null);
  const draggingRef = useRef(false);
  const gestureModeRef = useRef(false);
  const pointersRef = useRef(new Map<number, { x: number; y: number }>());
  const pinchStartRef = useRef<{
    distance: number;
    scale: number;
    panX: number;
    panY: number;
    centroidX: number;
    centroidY: number;
  } | null>(null);
  const lastPointerTypeRef = useRef<string>("mouse");

  if (!draggingRef.current) {
    annotationsRef.current = annotations;
  }

  const applyViewTransform = useCallback((next: ViewTransform) => {
    viewRef.current = next;
    setViewTransform(next);
    const { offsetX, offsetY, drawWidth, drawHeight } = layoutRef.current;
    if (drawWidth <= 0 || drawHeight <= 0) return;
    const cx = offsetX + drawWidth / 2;
    const cy = offsetY + drawHeight / 2;
    const w = drawWidth * next.scale;
    const h = drawHeight * next.scale;
    const style = {
      left: cx - w / 2 + next.panX,
      top: cy - h / 2 + next.panY,
      width: w,
      height: h,
    };
    letterboxStyleRef.current = style;
    setLetterboxStyle(style);
  }, []);

  const recomputeLetterboxFromLayout = useCallback(() => {
    applyViewTransform(viewRef.current);
  }, [applyViewTransform]);

  const imageToCanvas = useCallback((x: number, y: number) => {
    const { left, top, width, height } = letterboxStyleRef.current;
    return { x: left + x * width, y: top + y * height };
  }, []);

  const canvasToImage = useCallback((x: number, y: number) => {
    const { left, top, width, height } = letterboxStyleRef.current;
    if (width <= 0 || height <= 0) return { x: 0, y: 0 };
    return {
      x: clamp01((x - left) / width),
      y: clamp01((y - top) / height),
    };
  }, []);

  const layoutCenter = useCallback(() => {
    const { offsetX, offsetY, drawWidth, drawHeight } = layoutRef.current;
    return { cx: offsetX + drawWidth / 2, cy: offsetY + drawHeight / 2 };
  }, []);

  const hitRadius = useCallback(() => {
    const base = lastPointerTypeRef.current === "touch" ? TOUCH_HIT_PX : MOUSE_HIT_PX;
    return base / Math.max(viewRef.current.scale, 0.001);
  }, []);

  const syncLayout = useCallback(() => {
    const img = imgRef.current;
    const container = containerRef.current;
    if (!img?.naturalWidth || !container) return;
    const rect = container.getBoundingClientRect();
    const scale = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight);
    const drawWidth = img.naturalWidth * scale;
    const drawHeight = img.naturalHeight * scale;
    const offsetX = (rect.width - drawWidth) / 2;
    const offsetY = (rect.height - drawHeight) / 2;
    layoutRef.current = { offsetX, offsetY, drawWidth, drawHeight };
    recomputeLetterboxFromLayout();
  }, [recomputeLetterboxFromLayout]);

  const drawOverlay = useCallback(() => {
    const canvas = overlayRef.current;
    const container = containerRef.current;
    const img = imgRef.current;
    if (!canvas || !container) return;

    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const pxW = Math.max(1, Math.floor(rect.width * dpr));
    const pxH = Math.max(1, Math.floor(rect.height * dpr));
    if (canvas.width !== pxW || canvas.height !== pxH) {
      canvas.width = pxW;
      canvas.height = pxH;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    if (!img?.naturalWidth) {
      ctx.fillStyle = "#aaa";
      ctx.fillText("Select or capture an image", 20, 30);
      return;
    }

    const radius = hitRadius();

    const drawAnn = (ann: AnnotationShape) => {
      const color = classColor(ann.class_id);
      const selected = ann.id === selectedId;
      ctx.lineWidth = selected ? 3 : 2;
      ctx.strokeStyle = color;
      ctx.fillStyle = `${color}33`;

      if (ann.type === "polygon" && ann.polygon.length >= 3) {
        ctx.beginPath();
        ann.polygon.forEach((p, idx) => {
          const c = imageToCanvas(p.x, p.y);
          if (idx === 0) ctx.moveTo(c.x, c.y);
          else ctx.lineTo(c.x, c.y);
        });
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ann.polygon.forEach((p) => {
          const c = imageToCanvas(p.x, p.y);
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(c.x, c.y, selected ? radius * 0.6 : radius * 0.45, 0, Math.PI * 2);
          ctx.fill();
        });
      } else if (ann.bbox) {
        const p1 = imageToCanvas(ann.bbox.x_min, ann.bbox.y_min);
        const p2 = imageToCanvas(ann.bbox.x_max, ann.bbox.y_max);
        ctx.fillRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
        ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
      }
    };

    annotationsRef.current.forEach(drawAnn);

    const tempBbox = tempBboxRef.current;
    if (tempBbox) {
      const p1 = imageToCanvas(tempBbox.x_min, tempBbox.y_min);
      const p2 = imageToCanvas(tempBbox.x_max, tempBbox.y_max);
      ctx.strokeStyle = "#fff";
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
      ctx.setLineDash([]);
    }

    const tempPoly = tempPolygonRef.current;
    if (tempPoly.length) {
      ctx.strokeStyle = "#fff";
      ctx.fillStyle = "#ffffff88";
      ctx.beginPath();
      tempPoly.forEach((p, idx) => {
        const c = imageToCanvas(p.x, p.y);
        if (idx === 0) ctx.moveTo(c.x, c.y);
        else ctx.lineTo(c.x, c.y);
      });
      ctx.stroke();
      tempPoly.forEach((p) => {
        const c = imageToCanvas(p.x, p.y);
        ctx.beginPath();
        ctx.arc(c.x, c.y, radius * 0.5, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }, [hitRadius, imageToCanvas, selectedId]);

  const scheduleOverlay = useCallback(() => {
    if (rafRef.current !== null) return;
    rafRef.current = window.requestAnimationFrame(() => {
      rafRef.current = null;
      drawOverlay();
    });
  }, [drawOverlay]);

  const hitTest = useCallback(
    (x: number, y: number) => {
      const radius = hitRadius();
      for (let i = annotationsRef.current.length - 1; i >= 0; i -= 1) {
        const ann = annotationsRef.current[i];
        if (ann.type === "polygon" && ann.polygon.length) {
          for (let v = 0; v < ann.polygon.length; v += 1) {
            const c = imageToCanvas(ann.polygon[v].x, ann.polygon[v].y);
            if (Math.hypot(c.x - x, c.y - y) < radius) {
              return { kind: "vertex" as const, ann, index: v };
            }
          }
        }
        if (ann.bbox) {
          const p1 = imageToCanvas(ann.bbox.x_min, ann.bbox.y_min);
          const p2 = imageToCanvas(ann.bbox.x_max, ann.bbox.y_max);
          if (x >= p1.x && x <= p2.x && y >= p1.y && y <= p2.y) {
            return { kind: "annotation" as const, ann };
          }
        }
      }
      return null;
    },
    [hitRadius, imageToCanvas],
  );

  const finishPolygon = useCallback(() => {
    if (tempPolygonRef.current.length < 3) return;
    onBeforeChange();
    const polygon = [...tempPolygonRef.current];
    const next = [
      ...annotationsRef.current,
      {
        id: uuid(),
        class_id: activeClassId,
        type: "polygon" as const,
        bbox: bboxFromPolygon(polygon),
        polygon,
      },
    ];
    tempPolygonRef.current = [];
    annotationsRef.current = next;
    onCommit(next);
    scheduleOverlay();
  }, [activeClassId, onCommit, onBeforeChange, scheduleOverlay]);

  const cancelPolygonDraft = useCallback(() => {
    tempPolygonRef.current = [];
    scheduleOverlay();
  }, [scheduleOverlay]);

  const undoLastPolygonPoint = useCallback(() => {
    if (!tempPolygonRef.current.length) return;
    tempPolygonRef.current = tempPolygonRef.current.slice(0, -1);
    scheduleOverlay();
  }, [scheduleOverlay]);

  const deleteSelected = useCallback(() => {
    if (!selectedId) return false;
    onBeforeChange();
    const next = annotationsRef.current.filter((a) => a.id !== selectedId);
    annotationsRef.current = next;
    onCommit(next);
    onSelect(null);
    scheduleOverlay();
    return true;
  }, [onCommit, onBeforeChange, onSelect, scheduleOverlay, selectedId]);

  const resetView = useCallback(() => {
    applyViewTransform({ scale: 1, panX: 0, panY: 0 });
    scheduleOverlay();
  }, [applyViewTransform, scheduleOverlay]);

  const getPolygonDraftLength = useCallback(() => tempPolygonRef.current.length, []);

  const handleImageLoad = useCallback(() => {
    resetView();
    syncLayout();
    scheduleOverlay();
  }, [resetView, scheduleOverlay, syncLayout]);

  useEffect(() => {
    scheduleOverlay();
  }, [annotations, selectedId, imageUrl, viewTransform, scheduleOverlay]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => {
      syncLayout();
      scheduleOverlay();
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [scheduleOverlay, syncLayout]);

  useEffect(() => {
    const overlay = overlayRef.current;
    const container = containerRef.current;
    if (!overlay || !container) return;

    const pointerPos = (event: PointerEvent) => {
      const rect = overlay.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    };

    const pointerDistance = (a: { x: number; y: number }, b: { x: number; y: number }) =>
      Math.hypot(a.x - b.x, a.y - b.y);

    const pointerCentroid = (points: { x: number; y: number }[]) => {
      const sum = points.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
      return { x: sum.x / points.length, y: sum.y / points.length };
    };

    const updatePinch = () => {
      const points = [...pointersRef.current.values()];
      if (points.length < 2 || !pinchStartRef.current) return;
      const start = pinchStartRef.current;
      const dist = pointerDistance(points[0], points[1]);
      const centroid = pointerCentroid(points);
      const nextScale = clampZoom(start.scale * (dist / start.distance));
      const { cx, cy } = layoutCenter();
      const scaleRatio = nextScale / start.scale;
      const panX = start.panX + (centroid.x - start.centroidX) - (start.centroidX - cx - start.panX) * (scaleRatio - 1);
      const panY = start.panY + (centroid.y - start.centroidY) - (start.centroidY - cy - start.panY) * (scaleRatio - 1);
      applyViewTransform({ scale: nextScale, panX, panY });
      scheduleOverlay();
    };

    const onPointerDown = (event: PointerEvent) => {
      lastPointerTypeRef.current = event.pointerType;
      const { x, y } = pointerPos(event);
      pointersRef.current.set(event.pointerId, { x, y });

      if (pointersRef.current.size >= 2) {
        gestureModeRef.current = true;
        dragModeRef.current = null;
        draggingRef.current = false;
        tempBboxRef.current = null;
        const points = [...pointersRef.current.values()];
        const view = viewRef.current;
        pinchStartRef.current = {
          distance: pointerDistance(points[0], points[1]),
          scale: view.scale,
          panX: view.panX,
          panY: view.panY,
          centroidX: pointerCentroid(points).x,
          centroidY: pointerCentroid(points).y,
        };
        overlay.setPointerCapture(event.pointerId);
        event.preventDefault();
        return;
      }

      const img = imgRef.current;
      if (!img?.naturalWidth) return;

      const imagePt = canvasToImage(x, y);
      const hit = hitTest(x, y);
      const zoomed = viewRef.current.scale > 1.01;

      if (hit?.kind === "vertex") {
        onBeforeChange();
        draggingRef.current = true;
        dragModeRef.current = "vertex";
        dragStartRef.current = { annId: hit.ann.id, vertexIndex: hit.index, x, y };
        onSelect(hit.ann.id);
        overlay.setPointerCapture(event.pointerId);
        event.preventDefault();
        scheduleOverlay();
        return;
      }
      if (hit?.kind === "annotation") {
        onBeforeChange();
        draggingRef.current = true;
        dragModeRef.current = "move";
        dragStartRef.current = {
          annId: hit.ann.id,
          x,
          y,
          orig: structuredClone(hit.ann),
        };
        onSelect(hit.ann.id);
        overlay.setPointerCapture(event.pointerId);
        event.preventDefault();
        scheduleOverlay();
        return;
      }

      if (zoomed && !hit) {
        draggingRef.current = true;
        dragModeRef.current = "pan-view";
        dragStartRef.current = {
          x,
          y,
          panX: viewRef.current.panX,
          panY: viewRef.current.panY,
        };
        overlay.setPointerCapture(event.pointerId);
        event.preventDefault();
        onSelect(null);
        return;
      }

      onSelect(null);
      if (activeTool === "bbox") {
        draggingRef.current = true;
        dragModeRef.current = "create-bbox";
        dragStartRef.current = imagePt;
        tempBboxRef.current = bboxFromPoints(imagePt.x, imagePt.y, imagePt.x, imagePt.y);
        overlay.setPointerCapture(event.pointerId);
      } else {
        tempPolygonRef.current = [...tempPolygonRef.current, imagePt];
      }
      event.preventDefault();
      scheduleOverlay();
    };

    const onPointerMove = (event: PointerEvent) => {
      if (!pointersRef.current.has(event.pointerId)) return;
      const { x, y } = pointerPos(event);
      pointersRef.current.set(event.pointerId, { x, y });

      if (gestureModeRef.current && pointersRef.current.size >= 2) {
        updatePinch();
        event.preventDefault();
        return;
      }

      if (!dragModeRef.current || !imgRef.current?.naturalWidth) return;
      const imagePt = canvasToImage(x, y);
      const mode = dragModeRef.current;
      const start = dragStartRef.current;

      if (mode === "pan-view" && start && "panX" in start) {
        applyViewTransform({
          scale: viewRef.current.scale,
          panX: (start.panX as number) + (x - (start.x as number)),
          panY: (start.panY as number) + (y - (start.y as number)),
        });
        scheduleOverlay();
        event.preventDefault();
        return;
      }

      if (mode === "create-bbox" && start && "x" in start && "y" in start) {
        tempBboxRef.current = bboxFromPoints(
          start.x as number,
          start.y as number,
          imagePt.x,
          imagePt.y,
        );
        scheduleOverlay();
        return;
      }

      if (mode === "move" && start && "orig" in start) {
        const orig = start.orig as AnnotationShape;
        const { width, height } = letterboxStyleRef.current;
        const dx = (x - (start.x as number)) / width;
        const dy = (y - (start.y as number)) / height;
        annotationsRef.current = annotationsRef.current.map((ann) => {
          if (ann.id !== start.annId) return ann;
          const moved: AnnotationShape = {
            ...ann,
            bbox: {
              x_min: clamp01(orig.bbox.x_min + dx),
              y_min: clamp01(orig.bbox.y_min + dy),
              x_max: clamp01(orig.bbox.x_max + dx),
              y_max: clamp01(orig.bbox.y_max + dy),
            },
          };
          if (ann.type === "polygon") {
            moved.polygon = orig.polygon.map((p) => ({
              x: clamp01(p.x + dx),
              y: clamp01(p.y + dy),
            }));
          }
          return moved;
        });
        scheduleOverlay();
        return;
      }

      if (mode === "vertex" && start && "vertexIndex" in start) {
        annotationsRef.current = annotationsRef.current.map((ann) => {
          if (ann.id !== start.annId || ann.type !== "polygon") return ann;
          const polygon = [...ann.polygon];
          polygon[start.vertexIndex as number] = imagePt;
          return { ...ann, polygon, bbox: bboxFromPolygon(polygon) };
        });
        scheduleOverlay();
      }
    };

    const onPointerUp = (event: PointerEvent) => {
      pointersRef.current.delete(event.pointerId);
      if (pointersRef.current.size < 2) {
        gestureModeRef.current = false;
        pinchStartRef.current = null;
      }
      if (pointersRef.current.size >= 2) {
        const points = [...pointersRef.current.values()];
        const view = viewRef.current;
        pinchStartRef.current = {
          distance: pointerDistance(points[0], points[1]),
          scale: view.scale,
          panX: view.panX,
          panY: view.panY,
          centroidX: pointerCentroid(points).x,
          centroidY: pointerCentroid(points).y,
        };
        return;
      }

      const mode = dragModeRef.current;
      if (mode === "create-bbox" && tempBboxRef.current) {
        const box = tempBboxRef.current;
        const width = box.x_max - box.x_min;
        const height = box.y_max - box.y_min;
        if (width > 0.005 && height > 0.005) {
          onBeforeChange();
          const next = [
            ...annotationsRef.current,
            {
              id: uuid(),
              class_id: activeClassId,
              type: "bbox" as const,
              bbox: box,
              polygon: [],
            },
          ];
          annotationsRef.current = next;
          onCommit(next);
        }
        tempBboxRef.current = null;
      } else if (mode === "move" || mode === "vertex") {
        onCommit([...annotationsRef.current]);
      }

      draggingRef.current = false;
      dragModeRef.current = null;
      dragStartRef.current = null;
      scheduleOverlay();
    };

    const onDoubleClick = (event: MouseEvent) => {
      if (activeTool !== "polygon" || tempPolygonRef.current.length < 3) return;
      event.preventDefault();
      finishPolygon();
    };

    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = overlay.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const view = viewRef.current;
      const delta = event.deltaY > 0 ? 0.9 : 1.1;
      const nextScale = clampZoom(view.scale * delta);
      const { cx, cy } = layoutCenter();
      const scaleRatio = nextScale / view.scale;
      const panX = view.panX + (x - cx - view.panX) * (1 - scaleRatio);
      const panY = view.panY + (y - cy - view.panY) * (1 - scaleRatio);
      applyViewTransform({ scale: nextScale, panX, panY });
      scheduleOverlay();
    };

    overlay.addEventListener("pointerdown", onPointerDown);
    overlay.addEventListener("pointermove", onPointerMove);
    overlay.addEventListener("pointerup", onPointerUp);
    overlay.addEventListener("pointercancel", onPointerUp);
    overlay.addEventListener("dblclick", onDoubleClick);
    container.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      overlay.removeEventListener("pointerdown", onPointerDown);
      overlay.removeEventListener("pointermove", onPointerMove);
      overlay.removeEventListener("pointerup", onPointerUp);
      overlay.removeEventListener("pointercancel", onPointerUp);
      overlay.removeEventListener("dblclick", onDoubleClick);
      container.removeEventListener("wheel", onWheel);
    };
  }, [
    activeClassId,
    activeTool,
    applyViewTransform,
    canvasToImage,
    finishPolygon,
    hitTest,
    layoutCenter,
    onCommit,
    onBeforeChange,
    onSelect,
    scheduleOverlay,
  ]);

  return {
    containerRef,
    imgRef,
    overlayRef,
    letterboxStyle,
    viewTransform,
    imageUrl,
    handleImageLoad,
    finishPolygon,
    cancelPolygonDraft,
    undoLastPolygonPoint,
    deleteSelected,
    resetView,
    getPolygonDraftLength,
  };
}
