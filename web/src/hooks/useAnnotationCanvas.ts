import { useCallback, useEffect, useRef } from "react";
import type { AnnotationShape, ActiveTool, BoundingBox, Point2D } from "../api/types";
import { bboxFromPoints, bboxFromPolygon, classColor, clamp01, uuid } from "../utils/geometry";

export interface CanvasLayout {
  offsetX: number;
  offsetY: number;
  drawWidth: number;
  drawHeight: number;
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

type DragMode = "create-bbox" | "move" | "vertex" | null;

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
  const dragModeRef = useRef<DragMode>(null);
  const dragStartRef = useRef<Record<string, unknown> | null>(null);
  const tempBboxRef = useRef<BoundingBox | null>(null);
  const tempPolygonRef = useRef<Point2D[]>([]);
  const annotationsRef = useRef(annotations);
  const rafRef = useRef<number | null>(null);
  const draggingRef = useRef(false);

  if (!draggingRef.current) {
    annotationsRef.current = annotations;
  }

  const imageToCanvas = useCallback((x: number, y: number) => {
    const { offsetX, offsetY, drawWidth, drawHeight } = layoutRef.current;
    return { x: offsetX + x * drawWidth, y: offsetY + y * drawHeight };
  }, []);

  const canvasToImage = useCallback((x: number, y: number) => {
    const { offsetX, offsetY, drawWidth, drawHeight } = layoutRef.current;
    return {
      x: clamp01((x - offsetX) / drawWidth),
      y: clamp01((y - offsetY) / drawHeight),
    };
  }, []);

  const syncLayout = useCallback(() => {
    const img = imgRef.current;
    const container = containerRef.current;
    if (!img?.naturalWidth || !container) return;
    const rect = container.getBoundingClientRect();
    const scale = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight);
    const drawWidth = img.naturalWidth * scale;
    const drawHeight = img.naturalHeight * scale;
    layoutRef.current = {
      offsetX: (rect.width - drawWidth) / 2,
      offsetY: (rect.height - drawHeight) / 2,
      drawWidth,
      drawHeight,
    };
    img.style.left = `${layoutRef.current.offsetX}px`;
    img.style.top = `${layoutRef.current.offsetY}px`;
    img.style.width = `${drawWidth}px`;
    img.style.height = `${drawHeight}px`;
  }, []);

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
          ctx.arc(c.x, c.y, selected ? 5 : 4, 0, Math.PI * 2);
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
      ctx.beginPath();
      tempPoly.forEach((p, idx) => {
        const c = imageToCanvas(p.x, p.y);
        if (idx === 0) ctx.moveTo(c.x, c.y);
        else ctx.lineTo(c.x, c.y);
      });
      ctx.stroke();
    }
  }, [imageToCanvas, selectedId]);

  const scheduleOverlay = useCallback(() => {
    if (rafRef.current !== null) return;
    rafRef.current = window.requestAnimationFrame(() => {
      rafRef.current = null;
      drawOverlay();
    });
  }, [drawOverlay]);

  const hitTest = useCallback(
    (x: number, y: number) => {
      for (let i = annotationsRef.current.length - 1; i >= 0; i -= 1) {
        const ann = annotationsRef.current[i];
        if (ann.type === "polygon" && ann.polygon.length) {
          for (let v = 0; v < ann.polygon.length; v += 1) {
            const c = imageToCanvas(ann.polygon[v].x, ann.polygon[v].y);
            if (Math.hypot(c.x - x, c.y - y) < 8) return { kind: "vertex" as const, ann, index: v };
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
    [imageToCanvas],
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

  const handleImageLoad = useCallback(() => {
    syncLayout();
    scheduleOverlay();
  }, [scheduleOverlay, syncLayout]);

  useEffect(() => {
    scheduleOverlay();
  }, [annotations, selectedId, imageUrl, scheduleOverlay]);

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
    if (!overlay) return;

    const pointerPos = (event: MouseEvent) => {
      const rect = overlay.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    };

    const onMouseDown = (event: MouseEvent) => {
      const img = imgRef.current;
      if (!img?.naturalWidth) return;
      const { x, y } = pointerPos(event);
      const imagePt = canvasToImage(x, y);
      const hit = hitTest(x, y);

      if (hit?.kind === "vertex") {
        onBeforeChange();
        draggingRef.current = true;
        dragModeRef.current = "vertex";
        dragStartRef.current = { annId: hit.ann.id, vertexIndex: hit.index, x, y };
        onSelect(hit.ann.id);
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
        scheduleOverlay();
        return;
      }

      onSelect(null);
      if (activeTool === "bbox") {
        draggingRef.current = true;
        dragModeRef.current = "create-bbox";
        dragStartRef.current = imagePt;
        tempBboxRef.current = bboxFromPoints(imagePt.x, imagePt.y, imagePt.x, imagePt.y);
      } else {
        tempPolygonRef.current = [...tempPolygonRef.current, imagePt];
      }
      scheduleOverlay();
    };

    const onMouseMove = (event: MouseEvent) => {
      if (!dragModeRef.current || !imgRef.current?.naturalWidth) return;
      const { x, y } = pointerPos(event);
      const imagePt = canvasToImage(x, y);
      const mode = dragModeRef.current;
      const start = dragStartRef.current;

      if (mode === "create-bbox" && start && "x" in start && "y" in start) {
        tempBboxRef.current = bboxFromPoints(start.x as number, start.y as number, imagePt.x, imagePt.y);
        scheduleOverlay();
        return;
      }

      if (mode === "move" && start && "orig" in start) {
        const orig = start.orig as AnnotationShape;
        const dx = (x - (start.x as number)) / layoutRef.current.drawWidth;
        const dy = (y - (start.y as number)) / layoutRef.current.drawHeight;
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

    const onMouseUp = () => {
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

    overlay.addEventListener("mousedown", onMouseDown);
    overlay.addEventListener("mousemove", onMouseMove);
    overlay.addEventListener("mouseup", onMouseUp);
    overlay.addEventListener("dblclick", onDoubleClick);
    return () => {
      overlay.removeEventListener("mousedown", onMouseDown);
      overlay.removeEventListener("mousemove", onMouseMove);
      overlay.removeEventListener("mouseup", onMouseUp);
      overlay.removeEventListener("dblclick", onDoubleClick);
    };
  }, [
    activeClassId,
    activeTool,
    canvasToImage,
    finishPolygon,
    hitTest,
    onCommit,
    onBeforeChange,
    onSelect,
    scheduleOverlay,
  ]);

  return {
    containerRef,
    imgRef,
    overlayRef,
    imageUrl,
    handleImageLoad,
    finishPolygon,
    cancelPolygonDraft,
    deleteSelected,
  };
}
