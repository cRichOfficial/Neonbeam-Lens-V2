import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { ActiveTool, AnnotationShape } from "../api/types";
import { classColor } from "../utils/geometry";
import { useToast } from "../hooks/useToast";
import { AnnotationCanvas } from "./AnnotationCanvas";

interface AnnotateTabProps {
  active: boolean;
  selectedImageId: string | null;
  onSelectImage: (id: string | null) => void;
}

export function AnnotateTab({ active, selectedImageId, onSelectImage }: AnnotateTabProps) {
  const toast = useToast();
  const queryClient = useQueryClient();
  const canvasApiRef = useRef<{
    finishPolygon: () => void;
    cancelPolygonDraft: () => void;
    deleteSelected: () => boolean;
  } | null>(null);

  const [activeTool, setActiveTool] = useState<ActiveTool>("bbox");
  const [activeClassId, setActiveClassId] = useState(0);
  const [annotations, setAnnotations] = useState<AnnotationShape[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [reviewed, setReviewed] = useState(false);
  const [saveStatus, setSaveStatus] = useState("");
  const [newClassName, setNewClassName] = useState("");
  const undoStackRef = useRef<AnnotationShape[][]>([]);
  const saveTimerRef = useRef<number | null>(null);
  const annotationsRef = useRef(annotations);
  annotationsRef.current = annotations;

  const classesQuery = useQuery({
    queryKey: ["classes"],
    queryFn: api.getClasses,
  });

  const imagesQuery = useQuery({
    queryKey: ["images"],
    queryFn: () => api.listImages({ limit: 100 }),
    enabled: active,
  });

  const imageQuery = useQuery({
    queryKey: ["image", selectedImageId],
    queryFn: () => api.getImage(selectedImageId!),
    enabled: Boolean(selectedImageId),
  });

  useEffect(() => {
    if (imageQuery.data) {
      setAnnotations(imageQuery.data.annotations);
      setReviewed(imageQuery.data.reviewed);
      setSelectedId(null);
      undoStackRef.current = [];
      setSaveStatus("");
    }
  }, [imageQuery.data?.id, imageQuery.data]);

  const pushUndo = useCallback(() => {
    undoStackRef.current.push(structuredClone(annotationsRef.current));
    if (undoStackRef.current.length > 50) undoStackRef.current.shift();
  }, []);

  const scheduleSave = useCallback(
    (next: AnnotationShape[]) => {
      if (!selectedImageId) return;
      if (saveTimerRef.current) window.clearTimeout(saveTimerRef.current);
      saveTimerRef.current = window.setTimeout(async () => {
        try {
          await api.saveAnnotations(selectedImageId, next);
          setSaveStatus("Saved");
          queryClient.invalidateQueries({ queryKey: ["images"] });
        } catch (err) {
          setSaveStatus("Save failed");
          toast(err instanceof Error ? err.message : "Save failed", true);
        }
      }, 500);
    },
    [queryClient, selectedImageId, toast],
  );

  const commitAnnotations = useCallback(
    (next: AnnotationShape[]) => {
      setAnnotations(next);
      scheduleSave(next);
    },
    [scheduleSave],
  );

  const saveMutation = useMutation({
    mutationFn: () => api.saveAnnotations(selectedImageId!, annotations),
    onSuccess: () => {
      setSaveStatus("Saved");
      queryClient.invalidateQueries({ queryKey: ["images"] });
    },
    onError: (err: Error) => {
      setSaveStatus("Save failed");
      toast(err.message, true);
    },
  });

  const classesMutation = useMutation({
    mutationFn: api.setClasses,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["classes"] }),
    onError: (err: Error) => toast(err.message, true),
  });

  const deleteMutation = useMutation({
    mutationFn: () => api.deleteImage(selectedImageId!),
    onSuccess: async () => {
      toast("Image deleted");
      onSelectImage(null);
      await queryClient.invalidateQueries({ queryKey: ["images"] });
    },
    onError: (err: Error) => toast(err.message, true),
  });

  const patchMutation = useMutation({
    mutationFn: (nextReviewed: boolean) =>
      api.patchImage(selectedImageId!, { reviewed: nextReviewed }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["images"] });
      await queryClient.invalidateQueries({ queryKey: ["datasetStats"] });
    },
  });

  const classes = classesQuery.data?.classes ?? [];
  const images = imagesQuery.data?.items ?? [];

  const navigateImage = (delta: number) => {
    if (!selectedImageId || !images.length) return;
    const idx = images.findIndex((i) => i.id === selectedImageId);
    const next = images[idx + delta];
    if (next) onSelectImage(next.id);
  };

  useEffect(() => {
    if (!active) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "b" || e.key === "B") setActiveTool("bbox");
      if (e.key === "p" || e.key === "P") setActiveTool("polygon");
      if (e.key === "Delete") canvasApiRef.current?.deleteSelected();
      if (e.ctrlKey && e.key.toLowerCase() === "z") {
        const prev = undoStackRef.current.pop();
        if (prev) {
          setAnnotations(prev);
          scheduleSave(prev);
        }
      }
      if (/^[1-9]$/.test(e.key)) {
        const idx = Number(e.key) - 1;
        if (idx < classes.length) setActiveClassId(idx);
      }
      if (e.key === "Enter" && activeTool === "polygon") canvasApiRef.current?.finishPolygon();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [active, activeTool, classes.length, scheduleSave]);

  const imageUrl = selectedImageId ? api.imageUrl(selectedImageId, "preview") : null;

  return (
    <div className="grid-2">
      <aside className="stack panel">
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
          <button
            type="button"
            onClick={() => {
              const name = newClassName.trim();
              if (!name) return;
              classesMutation.mutate([...classes, name]);
              setNewClassName("");
              toast("Class added");
            }}
          >
            Add
          </button>
          <button
            type="button"
            className="danger"
            onClick={() => {
              if (classes.length <= 1) {
                toast("At least one class is required", true);
                return;
              }
              const next = classes.filter((_, idx) => idx !== activeClassId);
              setActiveClassId(0);
              classesMutation.mutate(next);
            }}
          >
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
        </div>
        {activeTool === "polygon" && (
          <div className="row">
            <button type="button" onClick={() => canvasApiRef.current?.finishPolygon()}>
              Finish polygon (Enter)
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
              <button
                type="button"
                className="danger"
                onClick={() => {
                  pushUndo();
                  const next = annotations.filter((a) => a.id !== ann.id);
                  setAnnotations(next);
                  if (selectedId === ann.id) setSelectedId(null);
                  scheduleSave(next);
                }}
              >
                Delete
              </button>
            </div>
          ))}
        </div>

        <label>
          <input
            type="checkbox"
            checked={reviewed}
            onChange={(e) => {
              setReviewed(e.target.checked);
              patchMutation.mutate(e.target.checked);
            }}
            disabled={!selectedImageId}
          />{" "}
          Mark reviewed
        </label>
        <div className="row">
          <button
            type="button"
            className="primary"
            onClick={() => saveMutation.mutate()}
            disabled={!selectedImageId}
          >
            Save
          </button>
          <button
            type="button"
            className="danger"
            onClick={() => {
              if (!selectedImageId) return;
              if (confirm("Delete this image and all annotations?")) deleteMutation.mutate();
            }}
            disabled={!selectedImageId}
          >
            Delete image
          </button>
        </div>
        <span className="muted">{saveStatus}</span>
        <p className="muted">Shortcuts: B box, P polygon, Del delete, Ctrl+Z undo, 1-9 class</p>
      </aside>

      <AnnotationCanvas
        imageUrl={imageUrl}
        annotations={annotations}
        selectedId={selectedId}
        activeClassId={activeClassId}
        activeTool={activeTool}
        onCommit={commitAnnotations}
        onSelect={setSelectedId}
        onBeforeChange={pushUndo}
        canvasApiRef={canvasApiRef}
      />
    </div>
  );
}
