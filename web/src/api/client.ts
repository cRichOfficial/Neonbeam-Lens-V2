import type {
  CameraSettings,
  DatasetStatsResponse,
  ExportResponse,
  ExportStatusResponse,
  ImageListResponse,
  ImageRecord,
  ImageVariant,
} from "./types";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || JSON.stringify(payload);
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  if (response.status === 204) return null as T;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return response.json() as Promise<T>;
  return response as unknown as T;
}

export const api = {
  getClasses: () => request<{ classes: string[] }>("/api/v1/dataset/classes"),
  setClasses: (classes: string[]) =>
    request<{ classes: string[] }>("/api/v1/dataset/classes", {
      method: "PUT",
      body: JSON.stringify({ classes }),
    }),
  capture: () => request<ImageRecord>("/api/v1/dataset/capture", { method: "POST" }),
  listImages: (params: Record<string, string | number | boolean> = {}) => {
    const query = new URLSearchParams(
      Object.entries(params).map(([k, v]) => [k, String(v)]),
    ).toString();
    return request<ImageListResponse>(`/api/v1/dataset/images?${query}`);
  },
  getImage: (id: string) => request<ImageRecord>(`/api/v1/dataset/images/${id}`),
  imageUrl: (id: string, variant: ImageVariant = "full") =>
    `/api/v1/dataset/images/${id}/file?variant=${variant}`,
  saveAnnotations: (id: string, annotations: ImageRecord["annotations"]) =>
    request<ImageRecord>(`/api/v1/dataset/images/${id}/annotations`, {
      method: "PUT",
      body: JSON.stringify({ annotations }),
    }),
  patchImage: (id: string, payload: { reviewed?: boolean; notes?: string }) =>
    request<ImageRecord>(`/api/v1/dataset/images/${id}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deleteImage: (id: string) =>
    request<void>(`/api/v1/dataset/images/${id}`, { method: "DELETE" }),
  exportDatasets: (payload: { reviewed_only: boolean; seed?: number }) =>
    request<ExportResponse>("/api/v1/dataset/export", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  exportStatus: () => request<ExportStatusResponse>("/api/v1/dataset/export/status"),
  getStats: (reviewedOnly = true) =>
    request<DatasetStatsResponse>(
      `/api/v1/dataset/stats?reviewed_only=${reviewedOnly ? "true" : "false"}`,
    ),
  getCameraSettings: () => request<CameraSettings>("/api/v1/camera/settings"),
  setCameraSettings: (payload: { exposure_ms?: number; analogue_gain?: number }) =>
    request<CameraSettings>("/api/v1/camera/settings", {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
  cameraStreamUrl: () => `/api/v1/camera/stream?size=main&ts=${Date.now()}`,
  health: () => request<{ camera?: { mode?: string } }>("/health"),
};
