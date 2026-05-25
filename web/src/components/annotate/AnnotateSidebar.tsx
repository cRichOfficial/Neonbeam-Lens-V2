import type { ActiveTool, AnnotationShape, ImageSummary } from "../../api/types";
import { classColor } from "../../utils/geometry";
import type { CanvasApi } from "../AnnotationCanvas";

export interface AnnotateSidebarProps {
  images: ImageSummary[];
  selectedImageId: string | null;
  onSelectImage: (id: string | null) => void;
  navigateImage: (delta: number) => void;
  classes: string[];
  activeClassId: number;
  setActiveClassId: (id: number) => void;
  newClassName: string;
  setNewClassName: (name: string) => void;
  onAddClass: () => void;
  onRemoveClass: () => void;
  activeTool: ActiveTool;
  setActiveTool: (tool: ActiveTool) => void;
  canvasApiRef: React.RefObject<CanvasApi | null>;
  annotations: AnnotationShape[];
  selectedId: string | null;
  setSelectedId: (id: string | null) => void;
  onDeleteAnnotation: (id: string) => void;
  reviewed: boolean;
  onReviewedChange: (checked: boolean) => void;
  onSave: () => void;
  onDeleteImage: () => void;
  saveStatus: string;
  saveDisabled: boolean;
  compact?: boolean;
}

export function AnnotateSidebar({
  images,
  selectedImageId,
  onSelectImage,
  navigateImage,
  classes,
  activeClassId,
  setActiveClassId,
  newClassName,
  setNewClassName,
  onAddClass,
  onRemoveClass,
  activeTool,
  setActiveTool,
  canvasApiRef,
  annotations,
  selectedId,
  setSelectedId,
  onDeleteAnnotation,
  reviewed,
  onReviewedChange,
  onSave,
  onDeleteImage,
  saveStatus,
  saveDisabled,
  compact = false,
}: AnnotateSidebarProps) {
  return (
    <aside className={`stack panel annotate-sidebar ${compact ? "annotate-sidebar--compact" : ""}`}>
      <h2>Images</h2>
      <div className="row">
        <button type="button" onClick={() => navigateImage(-1)}>
          Prev
        </button>
        <button type="button" onClick={() => navigateImage(1)}>
          Next
        </button>
      </div>
      <div className="image-list">
        {images.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`image-item ${item.id === selectedImageId ? "active" : ""}`}
            onClick={() => onSelectImage(item.id)}
          >
            <span>
              {item.filename.slice(0, 8)}… ({item.annotation_count})
            </span>
            <span className={`badge ${item.reviewed ? "reviewed" : ""}`}>
              {item.reviewed ? "reviewed" : "draft"}
            </span>
          </button>
        ))}
      </div>

      <h2>Classes</h2>
      <div className="class-list">
        {classes.map((name, index) => (
          <button
            key={name}
            type="button"
            className={`class-item ${index === activeClassId ? "active" : ""}`}
            onClick={() => setActiveClassId(index)}
          >
            <span>
              <span className="class-dot" style={{ background: classColor(index) }} />
              {index + 1}. {name}
            </span>
          </button>
        ))}
      </div>
      <div className="row">
        <input
          placeholder="New class name"
          value={newClassName}
          onChange={(e) => setNewClassName(e.target.value)}
        />
        <button type="button" onClick={onAddClass}>
          Add
        </button>
        <button type="button" className="danger" onClick={onRemoveClass}>
          Remove
        </button>
      </div>

      <h2>Tools</h2>
      <div className="row">
        <button
          type="button"
          className={activeTool === "bbox" ? "active-tool" : ""}
          onClick={() => setActiveTool("bbox")}
        >
          Box (B)
        </button>
        <button
          type="button"
          className={activeTool === "polygon" ? "active-tool" : ""}
          onClick={() => setActiveTool("polygon")}
        >
          Polygon (P)
        </button>
        <button type="button" onClick={() => canvasApiRef.current?.resetView()}>
          Reset view
        </button>
      </div>
      {activeTool === "polygon" && (
        <div className="row">
          <button type="button" onClick={() => canvasApiRef.current?.finishPolygon()}>
            Finish polygon (Enter)
          </button>
          <button type="button" onClick={() => canvasApiRef.current?.undoLastPolygonPoint()}>
            Undo point
          </button>
          <button type="button" onClick={() => canvasApiRef.current?.cancelPolygonDraft()}>
            Cancel
          </button>
        </div>
      )}

      <h2>Annotations</h2>
      <div className="annotation-list">
        {annotations.map((ann, idx) => (
          <div
            key={ann.id}
            className={`annotation-item ${ann.id === selectedId ? "active" : ""}`}
          >
            <button type="button" className="link-btn" onClick={() => setSelectedId(ann.id)}>
              {idx + 1}. {classes[ann.class_id] ?? `class ${ann.class_id}`} ({ann.type})
            </button>
            <button type="button" className="danger" onClick={() => onDeleteAnnotation(ann.id)}>
              Delete
            </button>
          </div>
        ))}
      </div>

      <label>
        <input
          type="checkbox"
          checked={reviewed}
          onChange={(e) => onReviewedChange(e.target.checked)}
          disabled={saveDisabled}
        />{" "}
        Mark reviewed
      </label>
      <div className="row">
        <button type="button" className="primary" onClick={onSave} disabled={saveDisabled}>
          Save
        </button>
        <button type="button" className="danger" onClick={onDeleteImage} disabled={saveDisabled}>
          Delete image
        </button>
      </div>
      <span className="muted">{saveStatus}</span>
      {!compact && (
        <p className="muted">Shortcuts: B box, P polygon, Del delete, Ctrl+Z undo, 1-9 class</p>
      )}
    </aside>
  );
}
