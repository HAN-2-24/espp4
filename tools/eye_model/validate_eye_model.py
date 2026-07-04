#!/usr/bin/env python3
"""Validate the eye-state detector on PC before ESP-DL conversion.

This script is intentionally stricter than a quick visual check:

* positive samples must have the right class and a box overlapping the YOLO
  label by at least --min-iou.
* negative samples must produce no detections above --conf.

Use it before touching firmware assets.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class GroundTruth:
    class_id: int
    xyxy: tuple[float, float, float, float]


@dataclass
class Prediction:
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]


def import_ultralytics():
    local_config = Path(".codex_tmp") / "eye_model"
    ultra_base = local_config / "ultralytics"
    mpl_base = local_config / "matplotlib"
    ultra_base.mkdir(parents=True, exist_ok=True)
    mpl_base.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(ultra_base.resolve())
    os.environ["MPLCONFIGDIR"] = str(mpl_base.resolve())
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "ultralytics is required.\n"
            f"Install it in this environment:\n  {sys.executable} -m pip install ultralytics"
        ) from exc
    return YOLO


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyYAML is required.\n"
            f"Install it in this environment:\n  {sys.executable} -m pip install pyyaml"
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset_dir(data_yaml: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return data_yaml.parent / path


def normalize_names(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {0: "eye_open", 1: "eye_closed"}


def list_images(paths: Iterable[Path]) -> list[Path]:
    images: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
        elif path.is_dir():
            images.extend(
                p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
    return sorted(images)


def image_label_path(image_path: Path, image_root: Path, label_root: Path) -> Path:
    rel = image_path.relative_to(image_root)
    return (label_root / rel).with_suffix(".txt")


def read_ground_truths(label_path: Path, width: int, height: int) -> list[GroundTruth]:
    if not label_path.exists():
        return []

    truths: list[GroundTruth] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = parts[:5]
        class_id = int(float(cls))
        center_x = float(cx) * width
        center_y = float(cy) * height
        box_w = float(bw) * width
        box_h = float(bh) * height
        x1 = center_x - box_w / 2.0
        y1 = center_y - box_h / 2.0
        x2 = center_x + box_w / 2.0
        y2 = center_y + box_h / 2.0
        truths.append(GroundTruth(class_id, (x1, y1, x2, y2)))
    return truths


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def make_negative_images(out_dir: Path, size: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    images: dict[str, np.ndarray] = {}
    images["black"] = np.zeros((size, size, 3), dtype=np.uint8)
    images["white"] = np.full((size, size, 3), 255, dtype=np.uint8)
    images["gray"] = np.full((size, size, 3), 128, dtype=np.uint8)

    grad_x = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    images["gradient_x"] = cv2.merge((grad_x, grad_x, grad_x))
    grad_y = grad_x.T
    images["gradient_y"] = cv2.merge((grad_y, grad_y, grad_y))

    checker = ((np.indices((size, size)).sum(axis=0) // 16) % 2 * 255).astype(np.uint8)
    images["checker"] = cv2.merge((checker, checker, checker))

    images["noise"] = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    images["dark_noise"] = rng.integers(0, 48, (size, size, 3), dtype=np.uint8)
    images["low_contrast"] = rng.integers(96, 160, (size, size, 3), dtype=np.uint8)

    blocks = np.zeros((size, size, 3), dtype=np.uint8)
    colors = [(40, 80, 180), (180, 120, 40), (60, 160, 80), (180, 180, 180)]
    half = size // 2
    blocks[:half, :half] = colors[0]
    blocks[:half, half:] = colors[1]
    blocks[half:, :half] = colors[2]
    blocks[half:, half:] = colors[3]
    images["color_blocks"] = blocks

    text = np.full((size, size, 3), 220, dtype=np.uint8)
    cv2.putText(text, "NO EYE", (size // 8, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    images["text"] = text

    paths: list[Path] = []
    for name, image in images.items():
        path = out_dir / f"{name}.jpg"
        cv2.imwrite(str(path), image)
        paths.append(path)
    return paths


def predictions_from_result(result, conf_threshold: float) -> list[Prediction]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    predictions: list[Prediction] = []
    for idx in range(len(boxes)):
        conf = float(boxes.conf[idx].item())
        if conf < conf_threshold:
            continue
        xyxy = tuple(float(v) for v in boxes.xyxy[idx].tolist())
        predictions.append(
            Prediction(
                class_id=int(boxes.cls[idx].item()),
                confidence=conf,
                xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
            )
        )
    predictions.sort(key=lambda pred: pred.confidence, reverse=True)
    return predictions


def draw_box(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def annotate_image(
    image_path: Path,
    truths: list[GroundTruth],
    predictions: list[Prediction],
    names: dict[int, str],
    out_path: Path,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    for truth in truths:
        draw_box(image, truth.xyxy, (0, 180, 0), f"gt {names.get(truth.class_id, truth.class_id)}")
    for pred in predictions:
        name = names.get(pred.class_id, str(pred.class_id))
        draw_box(image, pred.xyxy, (0, 0, 255), f"pred {name} {pred.confidence:.2f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def validate_positive(
    image_path: Path,
    truths: list[GroundTruth],
    predictions: list[Prediction],
    min_iou: float,
) -> tuple[bool, str, float, Prediction | None, GroundTruth | None]:
    best_iou = 0.0
    best_pred: Prediction | None = predictions[0] if predictions else None
    best_truth: GroundTruth | None = truths[0] if truths else None

    if not truths:
        return False, "missing_label", 0.0, best_pred, best_truth
    if not predictions:
        return False, "no_detection", 0.0, None, best_truth

    class_match_iou = 0.0
    class_match_pred: Prediction | None = None
    class_match_truth: GroundTruth | None = None
    for pred in predictions:
        for truth in truths:
            iou = box_iou(pred.xyxy, truth.xyxy)
            if iou > best_iou:
                best_iou = iou
                best_pred = pred
                best_truth = truth
            if pred.class_id == truth.class_id and iou > class_match_iou:
                class_match_iou = iou
                class_match_pred = pred
                class_match_truth = truth

    if class_match_pred is None:
        return False, "wrong_class", best_iou, best_pred, best_truth
    if class_match_iou < min_iou:
        return False, "low_iou", class_match_iou, class_match_pred, class_match_truth
    return True, "ok", class_match_iou, class_match_pred, class_match_truth


def relative_or_name(path: Path, root: Path | None) -> str:
    if root is None:
        return path.name
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def run_validation(args: argparse.Namespace) -> int:
    YOLO = import_ultralytics()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir = out_dir / "annotated"
    rows: list[dict[str, object]] = []

    data_yaml = Path(args.dataset) if args.dataset else None
    names = {0: "eye_open", 1: "eye_closed"}
    positive_images: list[Path] = []
    image_root: Path | None = None
    label_root: Path | None = None

    if data_yaml:
        data = load_yaml(data_yaml)
        names = normalize_names(data.get("names", names))
        dataset_root = resolve_dataset_dir(data_yaml, data.get("path", data_yaml.parent))
        image_root = resolve_dataset_dir(data_yaml, dataset_root / data[args.split])
        label_root = Path(str(image_root).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
        if label_root == image_root:
            label_root = dataset_root / "labels" / args.split
        positive_images = list_images([image_root])
        if args.max_positive > 0:
            positive_images = positive_images[: args.max_positive]

    negative_inputs: list[Path] = []
    negative_dirs = [Path(p) for p in args.negative_dir]
    negative_inputs.extend(list_images(negative_dirs))
    if args.generated_negatives:
        negative_inputs.extend(make_negative_images(out_dir / "generated_negatives", args.imgsz))

    if not positive_images and not negative_inputs:
        raise SystemExit("nothing to validate: pass --dataset and/or --negative-dir")

    model = YOLO(str(weights))

    total_positive = 0
    passed_positive = 0
    total_negative = 0
    passed_negative = 0

    def infer_one(path: Path) -> list[Prediction]:
        result = model.predict(
            source=str(path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]
        return predictions_from_result(result, args.conf)

    for image_path in positive_images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        assert image_root is not None
        assert label_root is not None
        truths = read_ground_truths(image_label_path(image_path, image_root, label_root), image.shape[1], image.shape[0])
        predictions = infer_one(image_path)
        ok, reason, best_iou, best_pred, best_truth = validate_positive(
            image_path, truths, predictions, args.min_iou
        )
        total_positive += 1
        passed_positive += 1 if ok else 0

        pred_name = names.get(best_pred.class_id, str(best_pred.class_id)) if best_pred else ""
        gt_name = names.get(best_truth.class_id, str(best_truth.class_id)) if best_truth else ""
        pred_box = " ".join(f"{v:.1f}" for v in best_pred.xyxy) if best_pred else ""
        gt_box = " ".join(f"{v:.1f}" for v in best_truth.xyxy) if best_truth else ""
        rows.append(
            {
                "group": "positive",
                "image": relative_or_name(image_path, image_root),
                "pass": int(ok),
                "reason": reason,
                "expected": gt_name,
                "predicted": pred_name,
                "confidence": f"{best_pred.confidence:.4f}" if best_pred else "",
                "iou": f"{best_iou:.4f}",
                "gt_xyxy": gt_box,
                "pred_xyxy": pred_box,
                "detections": len(predictions),
            }
        )

        if args.save_all or not ok:
            out_path = annotation_dir / "positive" / image_path.name
            annotate_image(image_path, truths, predictions, names, out_path)

    for image_path in negative_inputs:
        predictions = infer_one(image_path)
        ok = len(predictions) == 0
        reason = "ok" if ok else "false_positive"
        total_negative += 1
        passed_negative += 1 if ok else 0
        best_pred = predictions[0] if predictions else None
        pred_name = names.get(best_pred.class_id, str(best_pred.class_id)) if best_pred else ""
        pred_box = " ".join(f"{v:.1f}" for v in best_pred.xyxy) if best_pred else ""
        rows.append(
            {
                "group": "negative",
                "image": str(image_path),
                "pass": int(ok),
                "reason": reason,
                "expected": "no_eye",
                "predicted": pred_name,
                "confidence": f"{best_pred.confidence:.4f}" if best_pred else "",
                "iou": "",
                "gt_xyxy": "",
                "pred_xyxy": pred_box,
                "detections": len(predictions),
            }
        )

        if args.save_all or not ok:
            out_path = annotation_dir / "negative" / image_path.name
            annotate_image(image_path, [], predictions, names, out_path)

    report_path = out_dir / "report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "image",
                "pass",
                "reason",
                "expected",
                "predicted",
                "confidence",
                "iou",
                "gt_xyxy",
                "pred_xyxy",
                "detections",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    positive_rate = passed_positive / total_positive if total_positive else 0.0
    negative_rate = passed_negative / total_negative if total_negative else 0.0
    print(f"[summary] weights={weights}")
    print(f"[summary] report={report_path}")
    print(f"[summary] annotated={annotation_dir}")
    print(f"[summary] positive pass={passed_positive}/{total_positive} ({positive_rate:.3f})")
    print(f"[summary] negative pass={passed_negative}/{total_negative} ({negative_rate:.3f})")

    failures = [row for row in rows if not int(row["pass"])]
    if failures:
        print(f"[summary] failures={len(failures)}")
        for row in failures[:20]:
            print(
                "[fail] "
                f"{row['group']} {row['reason']} image={row['image']} "
                f"expected={row['expected']} predicted={row['predicted']} "
                f"conf={row['confidence']} iou={row['iou']}"
            )
        return 2

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="YOLO .pt or .onnx model")
    parser.add_argument("--dataset", help="YOLO data.yaml for positive validation")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--negative-dir", action="append", default=[], help="folder with no-eye images")
    parser.add_argument("--generated-negatives", action="store_true", help="also test generated blank/noise/pattern negatives")
    parser.add_argument("--out", default="runs/eye_state_eval/pc_validate")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--max-positive", type=int, default=400, help="0 means all positives")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-all", action="store_true", help="save annotations for passed images too")
    args = parser.parse_args()
    return run_validation(args)


if __name__ == "__main__":
    raise SystemExit(main())
