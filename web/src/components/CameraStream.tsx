import { useEffect, useState } from "react";
import { api } from "../api/client";

interface CameraStreamProps {
  active: boolean;
}

export function CameraStream({ active }: CameraStreamProps) {
  const [src, setSrc] = useState("");

  useEffect(() => {
    if (active) {
      setSrc(api.cameraStreamUrl());
    } else {
      setSrc("");
    }
  }, [active]);

  return (
    <div className="preview-wrap">
      {active && src ? (
        <img src={src} alt="Camera stream" className="preview-stream" />
      ) : (
        <p className="muted">Camera stream paused (switch to Capture tab to view)</p>
      )}
    </div>
  );
}
