#!/usr/bin/env python3
"""Train a custom YOLOv8 detector for laser bed materials."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on laser bed dataset")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "dataset.yaml",
        help="Path to dataset.yaml",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Base model checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--name", default="laser-bed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
    )
    print(results)
    print(f"Best weights: {args.project / args.name / 'weights' / 'best.pt'}")
    print("For segmentation models use: --model yolov8n-seg.pt --data data/dataset/export/segmentation/dataset.yaml")


if __name__ == "__main__":
    main()
