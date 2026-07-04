# Near-eye dataset direction

The target deployment domain is not a full face image. The latest local
reference samples in `data/eye_state_negative` are very close, blurry RGB eye
crops:

- `屏幕截图 2026-06-22 095447.png`: 336x232
- `屏幕截图 2026-06-22 095436.png`: 331x253

These user images are target-domain references only. They must not be used for
training, validation, augmentation, hard-negative mining, threshold tuning, or
model selection.

## Required training target

The model must remain a single direct detector:

```text
input: camera frame
output: eye bbox + eye_open / eye_closed + confidence
```

No runtime face detector, face landmark model, cascade, hand ROI calibration, or
two-stage model is allowed unless explicitly re-approved.

## Dataset priority

### Primary candidates

1. TEyeD

   Head-mounted / eye-facing real-world eye imagery. It is the closest public
   dataset family to the project target domain because it contains near-eye
   frames plus pupil, iris, and eyelid annotations. Use it as the first choice
   if it can be acquired locally.

   Source: https://arxiv.org/abs/2102.02115

2. OpenEDS / OpenEDS2020

   VR headset eye-facing camera data with eye-region segmentation labels. Use
   it as the second primary source if access is granted/downloaded.

   Sources:

   - https://arxiv.org/abs/1905.03702
   - https://arxiv.org/abs/2005.03876

### Supplemental open-eye sources

These are useful for close-up/open-eye diversity, but they are not sufficient by
themselves because they do not provide reliable closed-eye state labels.

1. EyeDentify / EyeDentify++

   Webcam cropped eye images with pupil diameter metadata. Processing explicitly
   prunes blink frames, so treat this as open-eye support only.

   Source: https://vijulshah.github.io/eyedentify/

2. LPW

   Head-mounted eye-region videos with pupil labels in daily indoor/outdoor
   conditions. Treat as open-eye support unless closed/blink labels are found in
   the local package.

   Source: https://arxiv.org/abs/1511.05768

### Supplemental blink/closed-eye sources

These can add closed-eye/blink examples, but they are less aligned with the
latest close-up reference samples because they are face-camera datasets.

1. RT-BENE

   Real-world extracted eye patches with open/blink labels. This is the first
   public dataset in the current workspace that is both downloadable and close
   enough to the target eye-patch domain to use as a training base.

   Local files:

   - `data/raw/rt_bene/s007_noglasses_eyes.tar`
   - `data/raw/rt_bene/s012_noglasses_eyes.tar`
   - `data/eye_state_yolo_rt_bene_s007_s012_noeye_v1`

   Source: https://zenodo.org/records/3685316

2. mEBAL2

   RGB/NIR face-camera blink database with labels. If used, eye crops must be
   generated only for training-data preparation; no face/landmark path may be
   introduced at runtime.

   Source: https://arxiv.org/abs/2309.07880

3. Existing local MRL

   Keep as auxiliary closed/open support only. It is not enough to prove the
   target near-eye RGB domain.

## Conversion rule

For segmentation/landmark datasets, labels may be converted offline into the
direct detector format:

```text
class eye_open:
  bbox from visible eye structures, with visible pupil/iris/sclera support

class eye_closed:
  bbox from eyelid/eye-region annotation or blink label, with no visible
  pupil/iris/sclera requirement

no-eye negative:
  same-source frames/crops without a valid eye annotation, plus public non-eye
  close-up negatives
```

This conversion is only for dataset creation. Runtime must still be the single
direct detector outputting bbox + open/closed + confidence.

## Stop condition

Do not deploy another model as "formal" unless PC validation passes on:

- target-like close-up eye positives,
- closed-eye positives,
- no-eye negatives,
- and the latest user reference images as diagnostic-only examples.
