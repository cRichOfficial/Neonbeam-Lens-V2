import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { ActiveTool, AnnotationShape } from "../api/types";
import { useToast } from "../hooks/useToast";
import { MOBILE_BREAKPOINT, useMediaQuery } from "../hooks/useMediaQuery";
import { AnnotationCanvas, type CanvasApi } from "./AnnotationCanvas";
import { AnnotateSidebar } from "./annotate/AnnotateSidebar";
import { MobileAnnotateSheet } from "./annotate/MobileAnnotateSheet";
import { MobileAnnotateToolbar } from "./annotate/MobileAnnotateToolbar";

interface AnnotateTabProps {
  active: boolean;
  selectedImageId: string | null;
  onSelectImage: (id: string | null) => void;
}

export function AnnotateTab({ active, selectedImageId, onSelectImage }: AnnotateTabProps) {
  const toast = useToast();
  const queryClient = useQueryClient();
  const isMobile = useMediaQuery(MOBILE_BREAKPOINT);
  const canvasApiRef = useRef<CanvasApi | null>(null);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [polygonDraftLength, setPolygonDraftLength] = useState(0);

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
      setPolygonDraftLength(0);
    }
  }, [imageQuery.data?.id, imageQuery.data]);

  useEffect(() => {
    if (!active || activeTool !== "polygon") {
      setPolygonDraftLength(0);
      return;
    }
    const id = window.setInterval(() => {
      setPolygonDraftLength(canvasApiRef.current?.getPolygonDraftLength() ?? 0);
    }, 200);
    return () => window.clearInterval(id);
  }, [active, activeTool]);

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

  const handleUndo = () => {
    const prev = undoStackRef.current.pop();
    if (prev) {
      setAnnotations(prev);
      scheduleSave(prev);
    }
  };

  const handleDelete = () => {
    if (polygonDraftLength > 0) {
      canvasApiRef.current?.cancelPolygonDraft();
      setPolygonDraftLength(0);
      return;
    }
    canvasApiRef.current?.deleteSelected();
  };

  useEffect(() => {
    if (!active) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "b" || e.key === "B") setActiveTool("bbox");
      if (e.key === "p" || e.key === "P") setActiveTool("polygon");
      if (e.key === "Delete") canvasApiRef.current?.deleteSelected();
      if (e.ctrlKey && e.key.toLowerCase() === "z") handleUndo();
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

  const sidebarProps = {
    images,
    selectedImageId,
    onSelectImage,
    navigateImage,
    classes,
    activeClassId,
    setActiveClassId,
    newClassName,
    setNewClassName,
    onAddClass: () => {
      const name = newClassName.trim();
      if (!name) return;
      classesMutation.mutate([...classes, name]);
      setNewClassName("");
      toast("Class added");
    },
    onRemoveClass: () => {
      if (classes.length <= 1) {
        toast("At least one class is required", true);
        return;
      }
      const next = classes.filter((_, idx) => idx !== activeClassId);
      setActiveClassId(0);
      classesMutation.mutate(next);
    },
    activeTool,
    setActiveTool,
    canvasApiRef,
    annotations,
    selectedId,
    setSelectedId,
    onDeleteAnnotation: (id: string) => {
      pushUndo();
      const next = annotations.filter((a) => a.id !== id);
      setAnnotations(next);
      if (selectedId === id) setSelectedId(null);
      scheduleSave(next);
    },
    reviewed,
    onReviewedChange: (checked: boolean) => {
      setReviewed(checked);
      patchMutation.mutate(checked);
    },
    onSave: () => saveMutation.mutate(),
    onDeleteImage: () => {
      if (!selectedImageId) return;
      if (confirm("Delete this image and all annotations?")) deleteMutation.mutate();
    },
    saveStatus,
    saveDisabled: !selectedImageId,
  };

  return (
    <div className={`annotate-layout ${isMobile ? "annotate-layout--mobile" : ""}`}>
      {!isMobile && <AnnotateSidebar {...sidebarProps} />}

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

      {isMobile && (
        <>
          <MobileAnnotateToolbar
            activeTool={activeTool}
            setActiveTool={setActiveTool}
            classes={classes}
            activeClassId={activeClassId}
            onOpenSheet={() => setSheetOpen(true)}
            onPrev={() => navigateImage(-1)}
            onNext={() => navigateImage(1)}
            onUndo={handleUndo}
            onDelete={handleDelete}
            hasSelection={Boolean(selectedId)}
            polygonDraftLength={polygonDraftLength}
            canvasApiRef={canvasApiRef}
          />
          <MobileAnnotateSheet
            open={sheetOpen}
            onClose={() => setSheetOpen(false)}
            {...sidebarProps}
          />
        </>
      )}
    </div>
  );
}
