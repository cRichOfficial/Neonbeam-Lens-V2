import { useAnnotationCanvas } from "../hooks/useAnnotationCanvas";
import type { ActiveTool, AnnotationShape } from "../api/types";

interface CanvasApi {
  finishPolygon: () => void;
  cancelPolygonDraft: () => void;
  deleteSelected: () => boolean;
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
    handleImageLoad,
    finishPolygon,
    cancelPolygonDraft,
    deleteSelected,
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

  canvasApiRef.current = { finishPolygon, cancelPolygonDraft, deleteSelected };

  return (
    <div className="panel canvas-wrap" ref={containerRef}>
      {imageUrl ? (
        <img
          ref={imgRef}
          src={imageUrl}
          alt="Annotation target"
          className="canvas-still"
          onLoad={handleImageLoad}
          draggable={false}
        />
      ) : null}
      <canvas ref={overlayRef} className="canvas-overlay" />
    </div>
  );
}
