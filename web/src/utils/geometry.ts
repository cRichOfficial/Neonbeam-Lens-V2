import type { BoundingBox, Point2D } from "../api/types";

export function uuid(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `id-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

const COLORS = ["#4da3ff", "#7ee787", "#f2cc60", "#ff7b72", "#d2a8ff", "#79c0ff"];

export function classColor(classId: number): string {
  return COLORS[classId % COLORS.length];
}

export function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function bboxFromPoints(x1: number, y1: number, x2: number, y2: number): BoundingBox {
  return {
    x_min: clamp01(Math.min(x1, x2)),
    y_min: clamp01(Math.min(y1, y2)),
    x_max: clamp01(Math.max(x1, x2)),
    y_max: clamp01(Math.max(y1, y2)),
  };
}

export function bboxFromPolygon(points: Point2D[]): BoundingBox {
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  return {
    x_min: clamp01(Math.min(...xs)),
    y_min: clamp01(Math.min(...ys)),
    x_max: clamp01(Math.max(...xs)),
    y_max: clamp01(Math.max(...ys)),
  };
}
