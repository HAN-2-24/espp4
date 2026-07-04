# Eye-State Detector Training Pipeline

Goal: train a direct detector that outputs:

```text
eye bbox + eye_open / eye_closed + confidence
```

This path intentionally avoids full-face detection, pedestrian detection,
manual ROI fatigue logic, and gray/texture heuristics.

Hard constraints:

- The final model is a single direct eye-state detector:
  `camera frame -> eye bbox + eye_open / eye_closed + confidence`.
- It must support no-eye / no-detection frames.
- Do not use user-owned images for training, validation, augmentation,
  hard-negative mining, threshold tuning, or model selection.
- Do not replace this with a face detector, face landmark model, dlib/OpenCV
  cascade, EAR/texture/threshold heuristic, manual ROI calibration, or a
  two-model cascade unless that direction is explicitly re-approved.

## Model Contract

Classes:

```text
0 eye_open
1 eye_closed
```

Target model asset:

```text
components/helmet_drivers/models/esp32p4/eye_state_detect_s8_v1.espdl
```

The bootstrap model can be trained from public cropped-eye datasets by
synthesizing full-frame YOLO samples. That is only for proving the training,
ONNX, ESP-DL, and firmware integration pipeline. It is not enough to declare
the fatigue detector finished.

## Environment

Use the existing model environment when available:

```powershell
D:\software\anaconda\envs\helmet-model\python.exe -m pip install -r tools/eye_model/requirements.txt
```

Or create a separate Python environment for model work:

```powershell
conda create -n helmet-eye python=3.11 -y
conda activate helmet-eye
python -m pip install -r tools/eye_model/requirements.txt
```

`convert_to_espdl.py` also requires Espressif's ESP-PPQ package. Install it in
the same environment following the ESP-DL tooling instructions already used for
this project.

## Prepare YOLO Dataset From MRL

MRL Eye Dataset is the preferred bootstrap source because its filenames encode
open/closed eye state. To download, extract, parse labels, and synthesize a YOLO
detector dataset:

```powershell
D:\software\anaconda\envs\helmet-model\python.exe tools\eye_model\prepare_mrl_eye_yolo.py `
  --download `
  --out data/eye_state_yolo_mrl `
  --imgsz 320 `
  --samples-per-image 2 `
  --clean
```

If the zip is already downloaded or extracted:

```powershell
D:\software\anaconda\envs\helmet-model\python.exe tools\eye_model\prepare_mrl_eye_yolo.py `
  --mrl-root data/raw/mrlEyes_2018_01 `
  --out data/eye_state_yolo_mrl `
  --imgsz 320 `
  --samples-per-image 2 `
  --clean
```

## Prepare YOLO Dataset From Generic Crops

For any other public cropped-eye data, split only by eye state:

```text
data/raw/eye_crops/open/
data/raw/eye_crops/closed/
```

Generate synthetic full-frame detection samples:

```powershell
python tools/eye_model/prepare_synthetic_eye_yolo.py `
  --open-dir data/raw/eye_crops/open `
  --closed-dir data/raw/eye_crops/closed `
  --out data/eye_state_yolo `
  --imgsz 320 `
  --samples-per-image 3 `
  --clean
```

This writes:

```text
data/eye_state_yolo/data.yaml
data/eye_state_yolo/images/{train,val,test}/
data/eye_state_yolo/labels/{train,val,test}/
data/eye_state_yolo/metadata.csv
```

## Train And Export ONNX

Smoke-test training from a tiny YOLO config:

```powershell
D:\software\anaconda\envs\helmet-model\python.exe tools\eye_model\train_eye_yolo.py `
  --data data/eye_state_yolo_mrl/data.yaml `
  --model yolov8n.yaml `
  --imgsz 320 `
  --epochs 80 `
  --batch 16 `
  --device cpu `
  --export-onnx `
  --exist-ok
```

If a pretrained weight is available locally, pass it explicitly:

```powershell
D:\software\anaconda\envs\helmet-model\python.exe tools\eye_model\train_eye_yolo.py `
  --data data/eye_state_yolo_mrl/data.yaml `
  --model path/to/yolov8n.pt `
  --imgsz 320 `
  --epochs 80 `
  --batch 16 `
  --device 0 `
  --pretrained `
  --export-onnx
```

## Inspect ONNX

Before ESP-DL conversion and firmware postprocess work:

```powershell
D:\software\anaconda\envs\helmet-model\python.exe tools\eye_model\inspect_onnx.py `
  runs/eye_state/eye_state_detect_v0/weights/best.onnx `
  --infer-shapes
```

Record the output tensor name and shape. `eye_state_detector.cpp` must decode
that actual output layout, not an assumed one.

## PC Validate Before Firmware

Validate the PyTorch model on source test images and no-eye negatives before
conversion:

```powershell
D:\software\anaconda\envs\helmet-model\python.exe tools\eye_model\validate_eye_model.py `
  --weights runs\detect\runs\eye_state\eye_state_detect_source_full_e10\weights\best.pt `
  --dataset data\eye_state_yolo_mrl\data.yaml `
  --split test `
  --generated-negatives `
  --out runs\eye_state_eval\pc_validate_source_full_e10_pt `
  --imgsz 320 `
  --conf 0.25 `
  --min-iou 0.5
```

For real camera or ordinary-view negatives, put images in a directory and add:

```powershell
  --negative-dir data\eye_state_negatives
```

The output CSV is `report.csv`; failed images are saved under `annotated/`.
The model is not ready for ESP-DL conversion if negative images produce
`eye_open` or `eye_closed` boxes.

## Convert To ESP-DL

```powershell
D:\software\anaconda\envs\helmet-model\python.exe tools\eye_model\convert_to_espdl.py `
  --onnx runs/eye_state/eye_state_detect_v0/weights/best.onnx `
  --out components/helmet_drivers/models/esp32p4/eye_state_detect_s8_v1.espdl `
  --target esp32p4 `
  --bits 8 `
  --input-shape 1,3,320,320 `
  --calib-dir data/eye_state_yolo_mrl/images/train `
  --partition-csv partitions.csv `
  --fail-on-runtime-unsupported
```

Board finding on 2026-06-18:

- The YOLOv8-derived `eye_state_detect_wflw_resume_e8` export is not acceptable
  as a board model. The original export contains PER_CHANNEL quantization and
  `Swish`, both rejected by the current ESP-DL runtime.
- `convert_to_espdl.py --force-p4-per-tensor --fail-on-runtime-unsupported`
  can produce a temporary model that passes the static PER_CHANNEL/Swish scan,
  but that model still asserted during board-side ESP-DL graph construction near
  the YOLO head. Treat this as evidence that the current YOLOv8 graph is not a
  reliable ESP32-P4 runtime target.
- Next model work should use an architecture/export path that is compatible
  with the board runtime from the start, then pass PC source-data validation
  before flashing.

After conversion, run the normal firmware build. The component CMake will embed
the `.espdl` file automatically.

## Firmware Next Step

Once a concrete `.onnx` / `.espdl` exists:

1. Update `eye_state_detector.cpp` preprocessing for the model input layout.
2. Decode the model's real output tensor into bbox/class/confidence.
3. Add NMS and confidence thresholds.
4. Return `OPEN`, `CLOSED`, `NO_DETECTION`, or `LOW_CONFIDENCE`.
5. Let `helmet_vision` keep calculating PERCLOS/blink from detector output.

Do not publish fatigue state from synthetic-only confidence without real helmet
validation.
