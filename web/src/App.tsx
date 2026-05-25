import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "./api/client";
import type { ActiveTab } from "./api/types";
import { AnnotateTab } from "./components/AnnotateTab";
import { CaptureTab } from "./components/CaptureTab";
import { ExportTab } from "./components/ExportTab";
import { ToastProvider } from "./hooks/useToast";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 5000, retry: 1 },
  },
});

function AppShell() {
  const [tab, setTab] = useState<ActiveTab>("capture");
  const [selectedImageId, setSelectedImageId] = useState<string | null>(null);

  const classesQuery = useQuery({
    queryKey: ["classes"],
    queryFn: api.getClasses,
  });

  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30000,
  });

  const handleCaptured = (imageId: string) => {
    setSelectedImageId(imageId);
    setTab("annotate");
  };

  const cameraMode = healthQuery.data?.camera?.mode ?? "unknown";
  const classCount = classesQuery.data?.classes.length ?? 0;

  return (
    <>
      <header className="app-header">
        <h1>Laser Bed Annotation</h1>
        <div className="status-bar">
          Camera: {cameraMode} | Classes: {classCount}
        </div>
      </header>

      <nav className="tabs">
        {(["capture", "annotate", "export"] as ActiveTab[]).map((name) => (
          <button
            key={name}
            type="button"
            className={`tab-btn ${tab === name ? "active" : ""}`}
            onClick={() => setTab(name)}
          >
            {name.charAt(0).toUpperCase() + name.slice(1)}
          </button>
        ))}
      </nav>

      {tab === "capture" && (
        <section className="tab-content">
          <CaptureTab active onCaptured={handleCaptured} />
        </section>
      )}
      {tab === "annotate" && (
        <section className="tab-content">
          <AnnotateTab
            active
            selectedImageId={selectedImageId}
            onSelectImage={setSelectedImageId}
          />
        </section>
      )}
      {tab === "export" && (
        <section className="tab-content">
          <ExportTab active />
        </section>
      )}
    </>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        <AppShell />
      </ToastProvider>
    </QueryClientProvider>
  );
}
