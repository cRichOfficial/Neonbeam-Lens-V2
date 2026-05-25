export interface Point2D {
  x: number;
  y: number;
}

export interface BoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface AnnotationShape {
  id: string;
  class_id: number;
  type: "bbox" | "polygon";
  bbox: BoundingBox;
  polygon: Point2D[];
}

export interface ImageRecord {
  id: string;
  filename: string;
  width: number;
  height: number;
  captured_at: string;
  reviewed: boolean;
  notes: string;
  annotations: AnnotationShape[];
}

export interface ImageSummary {
  id: string;
  filename: string;
  width: number;
  height: number;
  captured_at: string;
  reviewed: boolean;
  annotation_count: number;
}

export interface ImageListResponse {
  total: number;
  items: ImageSummary[];
}

export interface DatasetStatsResponse {
  total_images: number;
  reviewed_images: number;
  annotated_images: number;
  class_counts: Record<string, number>;
  train_val_split: number;
}

export interface DatasetExportSummary {
  task: "detection" | "segmentation";
  train_images: number;
  val_images: number;
  output_path: string;
  dataset_yaml: string;
}

export interface ExportResponse {
  exported_at: string;
  reviewed_only: boolean;
  train_val_split: number;
  class_counts: Record<string, number>;
  detection: DatasetExportSummary;
  segmentation: DatasetExportSummary;
}

export interface ExportStatusResponse {
  last_export_at: string | null;
  last_export: ExportResponse | null;
}

export interface CameraSettings {
  exposure_ms: number;
  analogue_gain: number;
  mount_height_mm: number;
  main_resolution: number[];
  lores_resolution: number[];
  camera_available: boolean;
  camera_mode: string;
}

export type ImageVariant = "full" | "preview" | "thumb";

export type ActiveTab = "capture" | "annotate" | "export";
export type ActiveTool = "bbox" | "polygon";
