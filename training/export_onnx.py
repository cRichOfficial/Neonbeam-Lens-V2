#!/usr/bin/env python3
"""Export trained YOLOv8 model to ONNX for Hailo compilation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    parser.add_argument("weights", type=Path, help="Path to best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=None, help="Output ONNX path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=True,
        opset=12,
        dynamic=False,
    )
    export_path = Path(export_path)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        export_path.replace(args.output)
        export_path = args.output
    print(f"Exported ONNX to {export_path}")
    print("Next: compile to HEF using training/compile_hef.md")


if __name__ == "__main__":
    main()
