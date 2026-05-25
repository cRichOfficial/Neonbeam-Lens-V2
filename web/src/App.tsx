import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "./api/client";
import type { ActiveTab } from "./api/types";
import { AnnotateTab } from "./components/AnnotateTab";
import { CaptureTab } from "./components/CaptureTab";
import { ExportTab } from "./components/ExportTab";
import { MOBILE_BREAKPOINT, useMediaQuery } from "./hooks/useMediaQuery";
import { ToastProvider } from "./hooks/useToast";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 5000, retry: 1 },
  },
});

const TABS: ActiveTab[] = ["capture", "annotate", "export"];

function AppShell() {
  const [tab, setTab] = useState<ActiveTab>("capture");
  const [selectedImageId, setSelectedImageId] = useState<string | null>(null);
  const isMobile = useMediaQuery(MOBILE_BREAKPOINT);

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
    <div className={`app-shell ${isMobile ? "app-shell--mobile" : ""}`}>
      <header className="app-header">
        <h1>Laser Bed Annotation</h1>
        <div className="status-bar">
          {isMobile ? `${classCount} classes` : `Camera: ${cameraMode} | Classes: ${classCount}`}
        </div>
      </header>

      {!isMobile && (
        <nav className="tabs tabs--top">
          {TABS.map((name) => (
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
      )}

      <main className="app-main">
        {tab === "capture" && (
          <section className="tab-content">
            <CaptureTab active onCaptured={handleCaptured} />
          </section>
        )}
        {tab === "annotate" && (
          <section className={`tab-content ${isMobile ? "tab-content--annotate-mobile" : ""}`}>
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
      </main>

      {isMobile && (
        <nav className="tabs tabs--bottom" aria-label="Main navigation">
          {TABS.map((name) => (
            <button
              key={name}
              type="button"
              className={`tab-btn tab-btn--bottom ${tab === name ? "active" : ""}`}
              onClick={() => setTab(name)}
            >
              {name.charAt(0).toUpperCase() + name.slice(1)}
            </button>
          ))}
        </nav>
      )}
    </div>
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
