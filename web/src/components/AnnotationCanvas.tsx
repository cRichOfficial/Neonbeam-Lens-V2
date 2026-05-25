import { useAnnotationCanvas } from "../hooks/useAnnotationCanvas";
import type { ActiveTool, AnnotationShape } from "../api/types";

export interface CanvasApi {
  finishPolygon: () => void;
  cancelPolygonDraft: () => void;
  undoLastPolygonPoint: () => void;
  deleteSelected: () => boolean;
  resetView: () => void;
  getPolygonDraftLength: () => number;
}

interface AnnotationCanvasProps {
  imageUrl: string | null;
  annotations: AnnotationShape[];
  selectedId: string | null;
  activeClassId: number;
  activeTool: ActiveTool;
  onCommit: (annotations: AnnotationShape[]) => void;
  onSelect: (id: string | null) => void;
  onBeforeChange: () => void;
  canvasApiRef: React.RefObject<CanvasApi | null>;
}

export function AnnotationCanvas({
  imageUrl,
  annotations,
  selectedId,
  activeClassId,
  activeTool,
  onCommit,
  onSelect,
  onBeforeChange,
  canvasApiRef,
}: AnnotationCanvasProps) {
  const {
    containerRef,
    imgRef,
    overlayRef,
    letterboxStyle,
    handleImageLoad,
    finishPolygon,
    cancelPolygonDraft,
    undoLastPolygonPoint,
    deleteSelected,
    resetView,
    getPolygonDraftLength,
  } = useAnnotationCanvas({
    imageUrl,
    annotations,
    selectedId,
    activeClassId,
    activeTool,
    onCommit,
    onSelect,
    onBeforeChange,
  });

  canvasApiRef.current = {
    finishPolygon,
    cancelPolygonDraft,
    undoLastPolygonPoint,
    deleteSelected,
    resetView,
    getPolygonDraftLength,
  };

  return (
    <div className="image-stage panel" ref={containerRef}>
      {imageUrl ? (
        <div
          className="image-stage__letterbox"
          style={{
            left: letterboxStyle.left,
            top: letterboxStyle.top,
            width: letterboxStyle.width,
            height: letterboxStyle.height,
          }}
        >
          <img
            ref={imgRef}
            src={imageUrl}
            alt="Annotation target"
            className="canvas-still"
            onLoad={handleImageLoad}
            draggable={false}
          />
        </div>
      ) : null}
      <canvas ref={overlayRef} className="canvas-overlay" />
    </div>
  );
}
