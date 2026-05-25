import type { ActiveTool } from "../../api/types";
import { classColor } from "../../utils/geometry";
import type { CanvasApi } from "../AnnotationCanvas";

interface MobileAnnotateToolbarProps {
  activeTool: ActiveTool;
  setActiveTool: (tool: ActiveTool) => void;
  classes: string[];
  activeClassId: number;
  onOpenSheet: () => void;
  onPrev: () => void;
  onNext: () => void;
  onUndo: () => void;
  onDelete: () => void;
  hasSelection: boolean;
  polygonDraftLength: number;
  canvasApiRef: React.RefObject<CanvasApi | null>;
}

export function MobileAnnotateToolbar({
  activeTool,
  setActiveTool,
  classes,
  activeClassId,
  onOpenSheet,
  onPrev,
  onNext,
  onUndo,
  onDelete,
  hasSelection,
  polygonDraftLength,
  canvasApiRef,
}: MobileAnnotateToolbarProps) {
  const className = classes[activeClassId] ?? `Class ${activeClassId + 1}`;

  return (
    <div className="mobile-annotate-toolbar">
      <div className="mobile-annotate-toolbar__row">
        <button
          type="button"
          className={`toolbar-btn ${activeTool === "bbox" ? "active-tool" : ""}`}
          onClick={() => setActiveTool("bbox")}
          aria-label="Box tool"
        >
          Box
        </button>
        <button
          type="button"
          className={`toolbar-btn ${activeTool === "polygon" ? "active-tool" : ""}`}
          onClick={() => setActiveTool("polygon")}
          aria-label="Polygon tool"
        >
          Poly
        </button>
        <button
          type="button"
          className="toolbar-btn toolbar-btn--class"
          onClick={onOpenSheet}
          aria-label="Select class"
        >
          <span className="class-dot" style={{ background: classColor(activeClassId) }} />
          {className}
        </button>
        <button type="button" className="toolbar-btn" onClick={onPrev} aria-label="Previous image">
          ◀
        </button>
        <button type="button" className="toolbar-btn" onClick={onNext} aria-label="Next image">
          ▶
        </button>
        {(hasSelection || polygonDraftLength > 0) && (
          <>
            <button type="button" className="toolbar-btn" onClick={onUndo} aria-label="Undo">
              Undo
            </button>
            <button type="button" className="toolbar-btn danger" onClick={onDelete} aria-label="Delete">
              Del
            </button>
          </>
        )}
        <button
          type="button"
          className="toolbar-btn"
          onClick={() => canvasApiRef.current?.resetView()}
          aria-label="Reset zoom"
        >
          Fit
        </button>
        <button type="button" className="toolbar-btn" onClick={onOpenSheet} aria-label="Menu">
          Menu
        </button>
      </div>
      {activeTool === "polygon" && polygonDraftLength > 0 && (
        <div className="mobile-annotate-toolbar__row mobile-annotate-toolbar__polygon">
          <span className="muted">{polygonDraftLength} point(s)</span>
          <button type="button" onClick={() => canvasApiRef.current?.undoLastPolygonPoint()}>
            Undo point
          </button>
          <button type="button" onClick={() => canvasApiRef.current?.finishPolygon()}>
            Finish
          </button>
          <button type="button" onClick={() => canvasApiRef.current?.cancelPolygonDraft()}>
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}
