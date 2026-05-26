# Hailo HEF Compilation Guide (Hailo-8L)

This project runs inference on a Raspberry Pi 5 with a Hailo-8L NPU. Custom YOLOv8 models must be compiled to `.hef` on an x86 development machine using the Hailo Dataflow Compiler (DFC).

## Prerequisites

- Ubuntu 22.04/24.04 or WSL2 on Windows
- Hailo Dataflow Compiler v3.x (from [Hailo Developer Zone](https://hailo.ai/developer-zone/sw-downloads/))
- Hailo Model Zoo v2.x branch (for Hailo-8 / Hailo-8L)
- Exported ONNX from `training/export_onnx.py`
- 100–200 representative bed images for calibration/quantization

## Workflow

### 1. Train locally

```bash
python training/train.py --data training/dataset.yaml --epochs 100
cp runs/train/laser-bed/weights/best.pt models/best.pt
```

### 2. Export ONNX

```bash
python training/export_onnx.py models/best.pt --output models/detection.onnx
```

### 3. Parse ONNX to HAR

```bash
hailomz parse --yaml path/to/yolov8n.yaml --ckpt models/detection.onnx --hw-arch hailo8l
```

Or use the Dataflow Compiler directly per your installed SDK version.

### 4. Optimize (quantize)

Provide a directory of calibration images captured from the laser bed:

```bash
hailomz optimize --har models/detection.har --calib-path data/dataset/images/train --hw-arch hailo8l
```

### 5. Compile to HEF

```bash
hailomz compile --har models/detection.har --hw-arch hailo8l --output-dir models/
```

### 6. Deploy to neonbeam-lens.richwerks.local

```bash
scp models/detection.hef crichards999@neonbeam-lens.richwerks.local:~/object-detection-v2/models/
ssh crichards999@neonbeam-lens.richwerks.local "sudo systemctl restart laser-detection"
```

Update `config/default.yaml` if the model filename differs.

## Notes

- YOLOv8 HEF models require host-side NMS post-processing if you compile custom detection HEFs for offline experiments (runtime API uses FastSAM only).
- For initial integration testing, the Pi ships with `/usr/share/hailo-models/yolov8s_h8l.hef` after `sudo apt install hailo-all`.
- Instance segmentation models (`yolov8n_seg`, etc.) need additional mask decoding on the CPU.

## References

- [Hailo Model Zoo HAILO8L object detection models](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst)
- [YOLOv8 to Hailo HEF pipeline example](https://github.com/arsatyants/hailo_model_generator)
