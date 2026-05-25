import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { useToast } from "../hooks/useToast";
import { CameraStream } from "./CameraStream";

interface CaptureTabProps {
  active: boolean;
  onCaptured: (imageId: string) => void;
}

export function CaptureTab({ active, onCaptured }: CaptureTabProps) {
  const toast = useToast();
  const queryClient = useQueryClient();
  const [exposure, setExposure] = useState<number | "">("");
  const [gain, setGain] = useState<number | "">("");

  const settingsQuery = useQuery({
    queryKey: ["cameraSettings"],
    queryFn: api.getCameraSettings,
  });

  const imagesQuery = useQuery({
    queryKey: ["images", "recent"],
    queryFn: () => api.listImages({ limit: 10 }),
  });

  const classesQuery = useQuery({
    queryKey: ["classes"],
    queryFn: api.getClasses,
  });

  useEffect(() => {
    if (settingsQuery.data) {
      setExposure(settingsQuery.data.exposure_ms);
      setGain(settingsQuery.data.analogue_gain);
    }
  }, [settingsQuery.data]);

  const captureMutation = useMutation({
    mutationFn: api.capture,
    onSuccess: async (record) => {
      toast("Image captured");
      await queryClient.invalidateQueries({ queryKey: ["images"] });
      onCaptured(record.id);
    },
    onError: (err: Error) => toast(err.message, true),
  });

  const settingsMutation = useMutation({
    mutationFn: api.setCameraSettings,
    onSuccess: () => toast("Camera settings updated"),
    onError: (err: Error) => toast(err.message, true),
  });

  const handleCapture = () => {
    const classes = classesQuery.data?.classes ?? [];
    if (!classes.length) {
      toast("Add at least one class in the Annotate tab first", true);
      return;
    }
    captureMutation.mutate();
  };

  return (
    <div className="grid-2">
      <div className="stack panel">
        <h2>Live Camera</h2>
        <CameraStream active={active} />
        <div className="row">
          <button className="primary" onClick={handleCapture} disabled={captureMutation.isPending}>
            Capture to Dataset
          </button>
        </div>
        <details className="collapsible">
          <summary>Camera settings</summary>
          <div className="stack">
            <label>
              Exposure (ms){" "}
              <input
                type="number"
                min={0.1}
                step={0.1}
                value={exposure}
                onChange={(e) => setExposure(Number(e.target.value))}
              />
            </label>
            <label>
              Gain{" "}
              <input
                type="number"
                min={0.1}
                step={0.1}
                value={gain}
                onChange={(e) => setGain(Number(e.target.value))}
              />
            </label>
            <button
              onClick={() =>
                settingsMutation.mutate({
                  exposure_ms: typeof exposure === "number" ? exposure : undefined,
                  analogue_gain: typeof gain === "number" ? gain : undefined,
                })
              }
            >
              Apply settings
            </button>
          </div>
        </details>
      </div>
      <div className="stack panel">
        <h2>Recent Captures</h2>
        <p className="muted">Click a thumbnail to annotate it.</p>
        <div className="thumb-strip">
          {(imagesQuery.data?.items ?? []).map((item) => (
            <button
              key={item.id}
              type="button"
              className="thumb"
              onClick={() => onCaptured(item.id)}
            >
              <img src={api.imageUrl(item.id, "thumb")} alt={item.filename} />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
