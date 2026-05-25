import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ExportResponse } from "../api/types";
import { useToast } from "../hooks/useToast";

interface ExportTabProps {
  active: boolean;
}

export function ExportTab({ active }: ExportTabProps) {
  const toast = useToast();
  const [reviewedOnly, setReviewedOnly] = useState(true);
  const [exportResult, setExportResult] = useState<ExportResponse | null>(null);

  const statsQuery = useQuery({
    queryKey: ["datasetStats", reviewedOnly],
    queryFn: () => api.getStats(reviewedOnly),
    enabled: active,
  });

  const exportMutation = useMutation({
    mutationFn: () => api.exportDatasets({ reviewed_only: reviewedOnly, seed: 42 }),
    onSuccess: (result) => {
      setExportResult(result);
      toast("Export complete");
    },
    onError: (err: Error) => toast(err.message, true),
  });

  useEffect(() => {
    if (!active) return;
    api.exportStatus()
      .then((status) => setExportResult(status.last_export))
      .catch(() => setExportResult(null));
  }, [active]);

  const split = statsQuery.data?.train_val_split ?? 0.8;
  const splitLabel = `${Math.round(split * 100)}% train / ${Math.round((1 - split) * 100)}% val`;

  return (
    <div className="grid-2">
      <div className="stack panel">
        <h2>Dataset Export</h2>
        <p className="muted">
          Exports both YOLO detection and segmentation datasets in one action. Box labels are
          converted to 4-corner polygons for segmentation export.
        </p>
        <label>
          <input
            type="checkbox"
            checked={reviewedOnly}
            onChange={(e) => setReviewedOnly(e.target.checked)}
          />{" "}
          Export reviewed images only
        </label>
        <p className="muted">Split: {splitLabel}</p>
        <button
          className="primary"
          onClick={() => exportMutation.mutate()}
          disabled={exportMutation.isPending}
        >
          Export datasets
        </button>
      </div>
      <div className="stack panel">
        <h2>Class counts (reviewed)</h2>
        <table className="stats">
          <thead>
            <tr>
              <th>Class</th>
              <th>Annotations</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(statsQuery.data?.class_counts ?? {}).map(([name, count]) => (
              <tr key={name}>
                <td>{name}</td>
                <td>{count}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div>
          {!exportResult ? (
            <p className="muted">No export run yet.</p>
          ) : (
            <>
              <p>Exported at {new Date(exportResult.exported_at).toLocaleString()}</p>
              <ul>
                <li>
                  Detection: {exportResult.detection.train_images} train /{" "}
                  {exportResult.detection.val_images} val →{" "}
                  <code>{exportResult.detection.dataset_yaml}</code>
                </li>
                <li>
                  Segmentation: {exportResult.segmentation.train_images} train /{" "}
                  {exportResult.segmentation.val_images} val →{" "}
                  <code>{exportResult.segmentation.dataset_yaml}</code>
                </li>
              </ul>
              <pre className="cmd">{`python training/train.py --data ${exportResult.detection.dataset_yaml}
python training/train.py --data ${exportResult.segmentation.dataset_yaml} --model yolov8n-seg.pt`}</pre>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
