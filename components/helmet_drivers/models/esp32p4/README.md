# Eye State Model

Place the ESP32-P4 ESP-DL model here:

```text
eye_state_detect_s8_v1.espdl
```

Current board model status:

- Active model route on 2026-06-23: three-class classifier.
- This supersedes the earlier direct-detector contract after the user explicitly
  approved dropping eye localization for now.
- Classes:
  - `0 eye_open`
  - `1 eye_closed`
  - `2 background`
- Source checkpoint:
  `runs/eye_state_classifier/eye_state_cls_capture_bg_v1/weights/best.pt`
- Source ONNX:
  `runs/eye_state_classifier/eye_state_cls_capture_bg_v1/weights/best.onnx`
- ESP-DL output:
  `components/helmet_drivers/models/esp32p4/eye_state_detect_s8_v1.espdl`
- ESP-DL size: `59,056` bytes.
- `eye_model` partition size: `3,407,872` bytes.
- SHA256:
  `9678C18B68395C3BA94598394A734E6C737656AB8563E39879CFD2FF16995472`
- Input/output contract:
  - input: `input 1x3x160x160`
  - output: `logits 1x3`
  - C++ postprocess applies softmax and maps `background` to `NO_DETECTION`.
- ONNX ops:
  `Conv`, `Relu`, `MaxPool`, `AveragePool`, `Flatten`, `Gemm`.
- Static conversion check:
  - opset 13
  - no negative axes
  - no `Swish`
  - no `DFL`
  - no graph `NMS` / `NonMax`
  - no `"PER_CHANNEL": true`
  - model fits the `eye_model` partition
- Training data:
  - `data/eye_state_capture/eye_open`
  - `data/eye_state_capture/eye_closed`
  - `data/public_no_eye_wflw_v2` for background only
  - `data/eye_state_negative` was not used for training
- Validation summary:
  - test `eye_open`: `15/15`
  - test `eye_closed`: `6/6`
  - test `background`: `33/45`
- Diagnostic-only user reference images:
  - `a03980d3895ae033d3c108e4858beb1c.jpg` -> `background=0.931`
  - `屏幕截图 2026-06-21 235918.png` -> `eye_closed=0.626`, `eye_open=0.369`
  - `屏幕截图 2026-06-22 095436.png` -> `eye_open=0.855`
  - `屏幕截图 2026-06-22 095447.png` -> `eye_open=0.952`

Historical direct-detector status:

- `eye_state_detect_s8_v1.espdl` was previously the deployed direct eye-state
  detector for the board.
- Source checkpoint:
  `runs/eye_state_direct/direct_eye_focus_lite10_balanced_v2/weights/best.pt`
- Source ONNX:
  `runs/eye_state_direct/direct_eye_focus_lite10_balanced_v2/weights/best_raw.onnx`
- ESP-DL output:
  `components/helmet_drivers/models/esp32p4/eye_state_detect_s8_v1.espdl`
- ESP-DL size: `40,880` bytes.
- `eye_model` partition size: `3,407,872` bytes.
- SHA256:
  `CBEAA8061BB3DC152F847D17D50F47CF31B23F3F0BB2B74519F049F6460E9231`
- Input/output contract:
  - input: `images 1x3x320x320`
  - output: `raw 1x7x20x20`
  - raw channels: `tx,ty,tw,th,obj,eye_open,eye_closed`
  - C++ postprocess decodes the fixed `20x20 = 400` grid into eye boxes and
    `eye_open` / `eye_closed` confidences
- Model architecture:
  - `depth=lite`, `width=10`
  - ONNX ops: `6 x Conv`, `5 x Relu`
- Static conversion check:
  - opset 13
  - no negative axes
  - no `Swish`
  - no ONNX decode subgraph (`Slice` / `Mul` / `Add` / `Reshape`) in the
    deployed model
  - no runtime-blocking `PER_CHANNEL` report from the conversion script
  - model fits the `eye_model` partition

Runtime note:

```text
The earlier decoded export loaded a postprocess graph into ESP-DL and crashed
during graph construction near /Mul_2. The deployed file is now the raw-head
export; decode is intentionally done in eye_state_detector.cpp.
```

PC validation summary for this deployed model:

```text
latest best-epoch focus val positive:  ~70.6%
latest best-epoch eye_open:            ~71.9%
latest best-epoch eye_closed:          ~48.6%
latest best-epoch no_eye:              ~85.0%
```

Board runtime summary:

```text
Latest COM3 reset log:
  eye-state ESP-DL model loaded from eye_model
  no NO_MODEL / assert / task_wdt / panic observed
  eye infer ~= 247-248 ms
  occasional spikes observed above 290 ms
```

CPU runtime config:

```text
CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ=360
CONFIG_PM_ENABLE is not set
CONFIG_ESP_FORCE_400MHZ_ON_REV_LESS_V3 is not set
```

Known limitation:

```text
The deployed model is now the board-runnable small direct detector. It is a
deployment baseline, not the final fatigue-grade model; tiny closed-eye cases
still need more PC-side work before the alarm logic should trust them.
```

This limitation is caused by the current fixed `20x20 = 400` grid and the very
small full-frame WFLW closed-eye boxes after resizing to 320. The deployed model
is therefore the current board model, but full-frame tiny-eye performance will
need a higher-resolution or multi-scale model contract in a later revision.

Historical note: the previous YOLOv8-derived
`eye_state_detect_wflw_resume_e8` export was rejected. The board runtime failed
on `PER_CHANNEL` quantization and `Swish`; after those were removed, the YOLOv8
head still asserted during ESP-DL graph construction near `/model.22/Div_1`.
That YOLO route is no longer the deployed model.

The detector expects a direct eye-state detection model that outputs an eye
bounding box, an `OPEN` or `CLOSED` class, and a confidence score. Full-face
detection models and pedestrian/person models are not used in this pipeline.

## Required Contract

Primary model route:

- One-stage detector: detect an eye/eye-region box and classify it as `OPEN` or
  `CLOSED` in the same model.

Rejected model routes:

- Full-face detection or face landmarks as the primary locator.
- Person/pedestrian detection.
- Two-model cascade unless explicitly re-approved.
- Eye classifier without eye localization.
- Eye/iris landmark model without direct `OPEN` / `CLOSED` output.
- Manual ROI calibration as the fatigue signal.
- Gray/texture heuristic without a trained eye-state model.

The C++ postprocess must be written against the chosen `.espdl` output layout.
Until this file exists and its outputs are known, runtime must report `NO_MODEL`
or `NO_DETECTION` and must not raise fatigue from stale or fake eye data.

## Candidate Review

Checked candidates on 2026-06-13:

- `Saadbs7/YOLOv8-Sleep-Detector-with-Text-to-Speech-Alert`
  <https://github.com/Saadbs7/YOLOv8-Sleep-Detector-with-Text-to-Speech-Alert>
  - YOLOv8 detect model with `Awake` / `Sleep` classes.
  - Validation image shows boxes around the eye/upper-face region.
  - Has `runs/detect/train/weights/best.pt` and `last.pt`.
  - Not ready to embed: no license detected, no `.onnx` or `.espdl`, and the
    training `config.yaml` referenced by `args.yaml` is not in the repo.
- `PINTO0309/OCEC` <https://github.com/PINTO0309/OCEC>
  - Very small ONNX `OPEN`/`CLOSED` eye classifier, input is cropped eye
    imagery.
  - Not a detector by itself. The demo uses a separate whole-body/eye locator,
    so this only fits the two-stage route if a lightweight eye/iris detector is
    selected separately.
- `PINTO0309/PINTO_model_zoo/192_open-closed-eye-0001`
  <https://github.com/PINTO0309/PINTO_model_zoo/tree/main/192_open-closed-eye-0001>
  - MIT licensed model-zoo entry for OpenVINO `open-closed-eye-0001`.
  - `convert_script.txt` uses `H=32`, `W=32`, `MODEL=open-closed-eye`, and an
    ONNX input model with output node `19`.
  - Useful as a cropped-eye `OPEN`/`CLOSED` classifier, but it does not locate
    the eye in the full camera frame.
- `PINTO0309/PINTO_model_zoo/049_iris_landmark`
  <https://github.com/PINTO0309/PINTO_model_zoo/tree/main/049_iris_landmark>
  - MIT licensed iris landmark model-zoo entry with conversion scripts.
  - Not a full-frame eye detector and not an `OPEN`/`CLOSED` classifier. It can
    only help after an eye crop already exists, or as a second stage paired with
    a separate eye/iris locator.
- `fadilahrahmadiah/yolov8-face-detection-web`
  <https://github.com/fadilahrahmadiah/yolov8-face-detection-web>
  - Full-frame YOLO-style ONNX detector. The app resizes camera frames to
    `640x640`, runs `face_classification.onnx`, and postprocesses class labels
    `["eye", "face", "lips", "nose"]`.
  - This matches the required eye-location stage if runtime filters only the
    `eye` class.
  - Not ready to embed yet: no license detected, model provenance/training data
    is unclear, and ESP-DL conversion/output compatibility still has to be
    verified.
- `KT313/eye_tracking` <https://github.com/KT313/eye_tracking>
  - Has eye pose ONNX files.
  - Reject as the main route: README states it first uses YOLOv8n-Face, and the
    eye model is trained on cropped face video. It also does not classify
    `OPEN` / `CLOSED`.
- `asifprotick10/OpenClosedEye`
  <https://github.com/asifprotick10/OpenClosedEye>
  - Reject: it uses dlib frontal face detection and 68-point face landmarks to
    crop eyes before classification.

Current decision: build a one-stage `eye_open` / `eye_closed` detector only.
The model must be proven on PC source data before board integration. User-owned
images are not used for training, validation, augmentation, hard-negative
mining, threshold tuning, or model selection.
