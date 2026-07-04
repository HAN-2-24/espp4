#!/usr/bin/env python3
"""Build a multiview eye-state YOLO dataset from an existing eye YOLO dataset.

The source dataset is expected to contain only:

    0 eye_open
    1 eye_closed

This script does not use user-owned images. It derives additional views from
the source public/WFLW-derived labels:

- full-frame copies
- close-up eye crops with relabeled boxes
- face-part no-eye crops that avoid all eye boxes
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "test")
CLASS_NAMES = {0: "eye_open", 1: "eye_closed"}


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


def resolve_path(data_yaml: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return data_yaml.parent / path


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def imread_color(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        data = np.empty((0,), dtype=np.uint8)
    if data.size:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    data.tofile(str(path))


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    labels: list[tuple[int, float, float, float, float]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = parts[:5]
        class_id = int(float(cls))
        if class_id in CLASS_NAMES:
            labels.append((class_id, float(cx), float(cy), float(bw), float(bh)))
    return labels


def labels_to_text(labels: list[tuple[int, float, float, float, float]]) -> str:
    return "".join(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for cls, cx, cy, bw, bh in labels)


def labels_to_xyxy(
    labels: list[tuple[int, float, float, float, float]],
    width: int,
    height: int,
) -> list[tuple[int, tuple[float, float, float, float]]]:
    boxes = []
    for class_id, cx, cy, bw, bh in labels:
        px = cx * width
        py = cy * height
        pw = bw * width
        ph = bh * height
        boxes.append((class_id, (px - pw / 2.0, py - ph / 2.0, px + pw / 2.0, py + ph / 2.0)))
    return boxes


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def crop_relabel(
    image: np.ndarray,
    labels: list[tuple[int, float, float, float, float]],
    crop_box: tuple[int, int, int, int],
    min_visible: float,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    x1, y1, x2, y2 = crop_box
    crop = image[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]
    if crop_w <= 0 or crop_h <= 0:
        return crop, []

    relabeled: list[tuple[int, float, float, float, float]] = []
    for class_id, box in labels_to_xyxy(labels, image.shape[1], image.shape[0]):
        bx1, by1, bx2, by2 = box
        ix1, iy1 = max(bx1, x1), max(by1, y1)
        ix2, iy2 = min(bx2, x2), min(by2, y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        box_area = max(1.0, (bx2 - bx1) * (by2 - by1))
        if inter / box_area < min_visible:
            continue
        cx = ((ix1 + ix2) * 0.5 - x1) / crop_w
        cy = ((iy1 + iy2) * 0.5 - y1) / crop_h
        bw = (ix2 - ix1) / crop_w
        bh = (iy2 - iy1) / crop_h
        relabeled.append((class_id, cx, cy, bw, bh))
    return crop, relabeled


def make_eye_closeup(
    image: np.ndarray,
    labels: list[tuple[int, float, float, float, float]],
    label_index: int,
    rng: random.Random,
    train: bool,
    min_size: int,
    scale_low: float,
    scale_high: float,
    height_scale: float,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]] | None:
    h, w = image.shape[:2]
    boxes = labels_to_xyxy(labels, w, h)
    class_id, box = boxes[label_index]
    x1, y1, x2, y2 = box
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    scale = rng.uniform(scale_low, scale_high) if train else (scale_low + scale_high) * 0.5
    crop_size = int(max(min_size, min(max(w, h), max(bw * scale, bh * scale * height_scale))))
    jitter_x = rng.uniform(-0.18, 0.18) * bw if train else 0.0
    jitter_y = rng.uniform(-0.20, 0.20) * bh if train else 0.0
    left = int(round(cx + jitter_x - crop_size * 0.5))
    top = int(round(cy + jitter_y - crop_size * 0.5))
    left = max(0, min(w - crop_size, left))
    top = max(0, min(h - crop_size, top))

    crop, relabeled = crop_relabel(image, labels, (left, top, left + crop_size, top + crop_size), 0.55)
    if not relabeled or not any(item[0] == class_id for item in relabeled):
        return None
    return crop, relabeled


def make_face_negative(
    image: np.ndarray,
    labels: list[tuple[int, float, float, float, float]],
    rng: random.Random,
) -> np.ndarray | None:
    h, w = image.shape[:2]
    boxes = [box for _class_id, box in labels_to_xyxy(labels, w, h)]
    if not boxes:
        return None
    eye_left = min(box[0] for box in boxes)
    eye_top = min(box[1] for box in boxes)
    eye_right = max(box[2] for box in boxes)
    eye_bottom = max(box[3] for box in boxes)
    eye_cx = (eye_left + eye_right) * 0.5

    for _ in range(32):
        kind = rng.choice(("lower", "upper", "side"))
        if kind == "lower":
            size = rng.randint(72, min(210, h))
            center_x = eye_cx + rng.uniform(-0.25, 0.25) * w
            center_y = rng.uniform(eye_bottom + size * 0.35, min(h - size * 0.35, eye_bottom + h * 0.45))
        elif kind == "upper":
            size = rng.randint(64, min(180, h))
            center_x = eye_cx + rng.uniform(-0.30, 0.30) * w
            center_y = rng.uniform(max(size * 0.5, eye_top - h * 0.28), max(size * 0.5, eye_top - size * 0.25))
        else:
            size = rng.randint(64, min(180, h))
            direction = -1.0 if rng.random() < 0.5 else 1.0
            center_x = eye_cx + direction * rng.uniform(w * 0.22, w * 0.38)
            center_y = rng.uniform(max(size * 0.5, eye_top), min(h - size * 0.5, eye_bottom + h * 0.25))

        left = int(round(center_x - size * 0.5))
        top = int(round(center_y - size * 0.5))
        left = max(0, min(w - size, left))
        top = max(0, min(h - size, top))
        crop_box = (float(left), float(top), float(left + size), float(top + size))
        if any(box_iou(crop_box, eye_box) > 0.0 for eye_box in boxes):
            continue
        crop = image[top : top + size, left : left + size]
        if crop.size:
            return crop
    return None


def write_data_yaml(out: Path) -> None:
    text = "\n".join(
        [
            f"path: {out.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: eye_open",
            "  1: eye_closed",
            "",
        ]
    )
    (out / "data.yaml").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--open-closeup-keep", type=float, default=0.45)
    parser.add_argument("--closed-closeups", type=int, default=2)
    parser.add_argument("--open-closeups", type=int, default=1)
    parser.add_argument("--macro-closeups", type=int, default=1)
    parser.add_argument("--face-negative-keep", type=float, default=0.75)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    cfg = load_yaml(data_yaml)
    root = resolve_path(data_yaml, cfg.get("path", data_yaml.parent))
    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    counts = {split: {"images": 0, "open": 0, "closed": 0, "neg": 0, "closeup": 0, "face_neg": 0} for split in SPLITS}
    meta_path = out / "metadata.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as meta:
        writer = csv.writer(meta)
        writer.writerow(["split", "image", "kind", "labels", "source"])

        for split in SPLITS:
            image_root = resolve_path(data_yaml, Path(root) / cfg[split])
            label_root = Path(str(image_root).replace("\\images\\", "\\labels\\"))
            for idx, src_image in enumerate(list_images(image_root)):
                rel = src_image.relative_to(image_root)
                src_label = (label_root / rel).with_suffix(".txt")
                labels = read_labels(src_label)
                image = imread_color(src_image)
                if image is None:
                    print(f"[warn] failed to read {src_image}", flush=True)
                    continue
                if image.shape[0] != args.imgsz or image.shape[1] != args.imgsz:
                    image = cv2.resize(image, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)

                stem = f"full_{idx:06d}"
                dst_image = out / "images" / split / f"{stem}.jpg"
                dst_label = out / "labels" / split / f"{stem}.txt"
                write_image(dst_image, image, args.jpeg_quality)
                dst_label.write_text(labels_to_text(labels), encoding="utf-8")
                writer.writerow([split, dst_image.as_posix(), "full", len(labels), src_image.as_posix()])
                counts[split]["images"] += 1
                if labels:
                    counts[split]["open"] += sum(1 for item in labels if item[0] == 0)
                    counts[split]["closed"] += sum(1 for item in labels if item[0] == 1)
                else:
                    counts[split]["neg"] += 1

                if not labels:
                    continue

                for label_idx, label in enumerate(labels):
                    class_id = label[0]
                    copies = args.closed_closeups if class_id == 1 else args.open_closeups
                    if split == "train" and class_id == 0 and rng.random() > args.open_closeup_keep:
                        continue
                    for copy_idx in range(copies):
                        closeup = make_eye_closeup(image, labels, label_idx, rng, split == "train", 64, 3.0, 5.8, 2.6)
                        if closeup is None:
                            continue
                        crop, crop_labels = closeup
                        crop = cv2.resize(crop, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
                        close_stem = f"closeup_{idx:06d}_{label_idx}_{copy_idx}"
                        close_image = out / "images" / split / f"{close_stem}.jpg"
                        close_label = out / "labels" / split / f"{close_stem}.txt"
                        write_image(close_image, crop, args.jpeg_quality)
                        close_label.write_text(labels_to_text(crop_labels), encoding="utf-8")
                        writer.writerow([split, close_image.as_posix(), "closeup", len(crop_labels), src_image.as_posix()])
                        counts[split]["images"] += 1
                        counts[split]["closeup"] += 1
                        counts[split]["open"] += sum(1 for item in crop_labels if item[0] == 0)
                        counts[split]["closed"] += sum(1 for item in crop_labels if item[0] == 1)

                    if split != "train" or class_id == 1 or rng.random() <= args.open_closeup_keep:
                        for macro_idx in range(args.macro_closeups):
                            macro = make_eye_closeup(image, labels, label_idx, rng, split == "train", 40, 1.9, 3.0, 1.8)
                            if macro is None:
                                continue
                            crop, crop_labels = macro
                            crop = cv2.resize(crop, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
                            macro_stem = f"macro_{idx:06d}_{label_idx}_{macro_idx}"
                            macro_image = out / "images" / split / f"{macro_stem}.jpg"
                            macro_label = out / "labels" / split / f"{macro_stem}.txt"
                            write_image(macro_image, crop, args.jpeg_quality)
                            macro_label.write_text(labels_to_text(crop_labels), encoding="utf-8")
                            writer.writerow([split, macro_image.as_posix(), "macro_closeup", len(crop_labels), src_image.as_posix()])
                            counts[split]["images"] += 1
                            counts[split]["closeup"] += 1
                            counts[split]["open"] += sum(1 for item in crop_labels if item[0] == 0)
                            counts[split]["closed"] += sum(1 for item in crop_labels if item[0] == 1)

                if split != "train" or rng.random() <= args.face_negative_keep:
                    neg = make_face_negative(image, labels, rng)
                    if neg is not None:
                        neg = cv2.resize(neg, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
                        neg_stem = f"face_neg_{idx:06d}"
                        neg_image = out / "images" / split / f"{neg_stem}.jpg"
                        neg_label = out / "labels" / split / f"{neg_stem}.txt"
                        write_image(neg_image, neg, args.jpeg_quality)
                        neg_label.write_text("", encoding="utf-8")
                        writer.writerow([split, neg_image.as_posix(), "face_negative", 0, src_image.as_posix()])
                        counts[split]["images"] += 1
                        counts[split]["neg"] += 1
                        counts[split]["face_neg"] += 1

    write_data_yaml(out)
    print(f"[info] wrote {out}", flush=True)
    for split in SPLITS:
        item = counts[split]
        print(
            f"[info] {split}: images={item['images']} open={item['open']} "
            f"closed={item['closed']} neg_images={item['neg']} "
            f"closeups={item['closeup']} face_negatives={item['face_neg']}",
            flush=True,
        )
    print(f"[info] metadata={meta_path}", flush=True)
    print(f"[info] data_yaml={out / 'data.yaml'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
