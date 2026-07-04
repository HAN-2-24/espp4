# Eye Model Data Use Policy

## Prohibited

`data/eye_state_negative` is prohibited for:

- training
- validation
- test-set closure
- augmentation
- hard-negative mining
- threshold tuning
- model selection
- deployment decisions

Do not convert images from this directory into YOLO labels. Do not merge a
dataset derived from this directory into any training dataset.

## Allowed Target-Domain Source

`data/eye_state_capture` may be used after manual bbox annotation.

Allowed manual annotation outputs:

- `data/eye_state_capture/manual_labels`
- `data/eye_state_capture/visualized_manual`

Allowed exported dataset:

- `data/eye_state_yolo_capture_manual_*`

## Invalidated Artifacts

The following artifacts are invalid because they include images derived from
`data/eye_state_negative`:

- `data/eye_state_yolo_user_standard_open_v1`
- `data/eye_state_yolo_mrl_capture_darkx50_standardx200_noeye_lite_v1`
- `runs/eye_state_direct/direct_eye_mrl_capture_darkx50_standardx200_noeye_lite_w8_grid20_ft10_v1`
