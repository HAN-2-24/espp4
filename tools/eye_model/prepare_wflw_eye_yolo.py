#!/usr/bin/env python3
"""Prepare a direct eye-open/eye-closed YOLO dataset from WFLW landmarks.

Runtime target:

    full frame -> eye_open / eye_closed boxes

The generated labels intentionally contain only eye boxes. WFLW face rectangles
are used only for optional sanity checks/negative crop avoidance and are never
written as model labels.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "opencv-python and numpy are required.\n"
        f"Install them in this environment:\n  {sys.executable} -m pip install opencv-python numpy"
    ) from exc


CLASSES = {"eye_open": 0, "eye_closed": 1}
LEFT_EYE = [60, 61, 62, 63, 64, 65, 66, 67]
RIGHT_EYE = [68, 69, 70, 71, 72, 73, 74, 75]
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class EyeBox:
    label: str
    xyxy: tuple[float, float, float, float]
    ear: float


@dataclass(frozen=True)
class Sample:
    split: str
    image_path: Path
    eyes: tuple[EyeBox, ...]


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_ear(points: list[tuple[float, float]], side: str) -> float:
    if side == "left":
        corner_a, corner_b = 60, 64
        pairs = ((61, 67), (62, 66), (63, 65))
    else:
        corner_a, corner_b = 68, 72
        pairs = ((69, 75), (70, 74), (71, 73))
    width = distance(points[corner_a], points[corner_b])
    if width <= 1e-6:
        return 0.0
    return sum(distance(points[a], points[b]) for a, b in pairs) / (3.0 * width)


def padded_box(
    points: list[tuple[float, float]],
    indices: list[int],
    image_w: int,
    image_h: int,
    pad_ratio: float,
) -> tuple[float, float, float, float]:
    xs = [points[idx][0] for idx in indices]
    ys = [points[idx][1] for idx in indices]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = width * pad_ratio
    pad_y = max(height * pad_ratio, width * pad_ratio * 0.45)
    return (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(image_w - 1), x2 + pad_x),
        min(float(image_h - 1), y2 + pad_y),
    )


def classify_ear(ear: float, closed_threshold: float, open_threshold: float) -> str | None:
    if ear <= closed_threshold:
        return "eye_closed"
    if ear >= open_threshold:
        return "eye_open"
    return None


def yolo_line(label: str, xyxy: tuple[float, float, float, float], image_w: int, image_h: int) -> str:
    x1, y1, x2, y2 = xyxy
    cx = ((x1 + x2) * 0.5) / image_w
    cy = ((y1 + y2) * 0.5) / image_h
    bw = (x2 - x1) / image_w
    bh = (y2 - y1) / image_h
    return f"{CLASSES[label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def parse_annotation_line(line: str) -> tuple[list[tuple[float, float]], str]:
    parts = line.strip().split()
    if len(parts) != 207:
        raise ValueError(f"expected 207 fields, got {len(parts)}")
    coords = [float(value) for value in parts[:196]]
    points = [(coords[idx * 2], coords[idx * 2 + 1]) for idx in range(98)]
    return points, parts[-1]


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image


def collect_samples(
    annotations_root: Path,
    images_root: Path,
    closed_threshold: float,
    open_threshold: float,
    pad_ratio: float,
) -> list[Sample]:
    samples: list[Sample] = []
    split_files = {
        "train": annotations_root / "list_98pt_rect_attr_train.txt",
        "test": annotations_root / "list_98pt_rect_attr_test.txt",
    }
    for src_split, path in split_files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            points, rel_image = parse_annotation_line(line)
            image_path = images_root / rel_image
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            image = read_image(image_path)
            h, w = image.shape[:2]

            eyes: list[EyeBox] = []
            for side, indices in (("left", LEFT_EYE), ("right", RIGHT_EYE)):
                ear = eye_ear(points, side)
                label = classify_ear(ear, closed_threshold, open_threshold)
                if label is None:
                    continue
                box = padded_box(points, indices, w, h, pad_ratio)
                x1, y1, x2, y2 = box
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue
                eyes.append(EyeBox(label, box, ear))

            if eyes:
                samples.append(Sample("test" if src_split == "test" else "train", image_path, tuple(eyes)))
    return samples


def assign_validation_split(samples: list[Sample], val_ratio: float, seed: int) -> list[Sample]:
    train = [sample for sample in samples if sample.split == "train"]
    test = [sample for sample in samples if sample.split == "test"]
    rng = random.Random(seed)
    rng.shuffle(train)
    val_count = int(len(train) * val_ratio)
    val = [Sample("val", sample.image_path, sample.eyes) for sample in train[:val_count]]
    train = [Sample("train", sample.image_path, sample.eyes) for sample in train[val_count:]]
    return train + val + test


def prepare_dirs(out_dir: Path, clean: bool) -> None:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml(out_dir: Path) -> None:
    text = "\n".join(
        [
            f"path: {out_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: eye_open",
            "  1: eye_closed",
            "",
        ]
    )
    (out_dir / "data.yaml").write_text(text, encoding="utf-8")


def copy_positive_samples(samples: list[Sample], out_dir: Path, writer: csv.writer) -> dict[str, dict[str, int]]:
    counts = {split: {"images": 0, "eye_open": 0, "eye_closed": 0} for split in SPLITS}
    seen_names = {split: set() for split in SPLITS}
    for sample in samples:
        split = sample.split
        stem = sample.image_path.stem
        name = sample.image_path.name
        if name in seen_names[split]:
            suffix = 1
            while f"{stem}_{suffix}{sample.image_path.suffix}" in seen_names[split]:
                suffix += 1
            name = f"{stem}_{suffix}{sample.image_path.suffix}"
        seen_names[split].add(name)

        image_out = out_dir / "images" / split / name
        label_out = (out_dir / "labels" / split / name).with_suffix(".txt")
        shutil.copyfile(sample.image_path, image_out)
        with label_out.open("w", encoding="utf-8") as f:
            image = read_image(sample.image_path)
            h, w = image.shape[:2]
            for eye in sample.eyes:
                f.write(yolo_line(eye.label, eye.xyxy, w, h))
                counts[split][eye.label] += 1
                writer.writerow(
                    [
                        split,
                        image_out.as_posix(),
                        eye.label,
                        f"{eye.ear:.6f}",
                        " ".join(f"{v:.2f}" for v in eye.xyxy),
                        sample.image_path.as_posix(),
                    ]
                )
        counts[split]["images"] += 1
    return counts


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


def make_generated_negative(rng: random.Random, size: int) -> np.ndarray:
    kind = rng.choice(("solid", "gradient", "noise", "checker", "rectangles", "lines"))
    if kind == "solid":
        return np.full((size, size, 3), rng.randint(0, 255), dtype=np.uint8)
    if kind == "gradient":
        a, b = rng.randint(0, 255), rng.randint(0, 255)
        line = np.linspace(a, b, size, dtype=np.float32)
        plane = np.tile(line, (size, 1)) if rng.random() < 0.5 else np.tile(line[:, None], (1, size))
        return np.dstack([np.clip(plane + rng.randint(-30, 30), 0, 255) for _ in range(3)]).astype(np.uint8)
    if kind == "noise":
        np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        low = rng.randint(0, 180)
        high = rng.randint(max(low + 1, 40), 256)
        return np_rng.integers(low, high, (size, size, 3), dtype=np.uint8)
    if kind == "checker":
        block = rng.randint(5, 40)
        checker = ((np.indices((size, size)).sum(axis=0) // block) % 2 * rng.randint(80, 255)).astype(np.uint8)
        return cv2.merge((checker, checker, checker))
    image = np.full((size, size, 3), rng.randint(25, 235), dtype=np.uint8)
    for _ in range(rng.randint(3, 18)):
        p1 = (rng.randint(0, size - 1), rng.randint(0, size - 1))
        p2 = (rng.randint(0, size - 1), rng.randint(0, size - 1))
        color = tuple(rng.randint(0, 255) for _ in range(3))
        if kind == "rectangles":
            cv2.rectangle(image, p1, p2, color, -1 if rng.random() < 0.6 else rng.randint(1, 4))
        else:
            cv2.line(image, p1, p2, color, rng.randint(1, 4))
    return image


def add_negative_samples(
    samples: list[Sample],
    out_dir: Path,
    rng: random.Random,
    generated_count: int,
    crop_count: int,
    negative_size: int,
    jpeg_quality: int,
    writer: csv.writer,
) -> dict[str, int]:
    counts = {split: 0 for split in SPLITS}
    split_samples = {split: [sample for sample in samples if sample.split == split] for split in SPLITS}

    for split in SPLITS:
        for idx in range(max(0, generated_count)):
            image = make_generated_negative(rng, negative_size)
            name = f"no_eye_generated_{idx:06d}.jpg"
            image_out = out_dir / "images" / split / name
            label_out = out_dir / "labels" / split / f"no_eye_generated_{idx:06d}.txt"
            cv2.imwrite(str(image_out), image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            label_out.write_text("", encoding="utf-8")
            writer.writerow([split, image_out.as_posix(), "no_eye", "", "", "generated"])
            counts[split] += 1

        candidates = split_samples[split]
        if not candidates:
            continue
        made = 0
        attempts = 0
        while made < crop_count and attempts < crop_count * 80:
            attempts += 1
            sample = rng.choice(candidates)
            image = read_image(sample.image_path)
            h, w = image.shape[:2]
            if h < 32 or w < 32:
                continue
            crop_w = rng.randint(max(24, w // 8), max(25, min(w, w // 2)))
            crop_h = rng.randint(max(24, h // 8), max(25, min(h, h // 2)))
            x1 = rng.randint(0, max(0, w - crop_w))
            y1 = rng.randint(0, max(0, h - crop_h))
            crop_box = (float(x1), float(y1), float(x1 + crop_w), float(y1 + crop_h))
            if any(box_iou(crop_box, eye.xyxy) > 0.001 for eye in sample.eyes):
                continue
            crop = image[y1:y1 + crop_h, x1:x1 + crop_w]
            if crop.size == 0:
                continue
            name = f"no_eye_crop_{made:06d}.jpg"
            image_out = out_dir / "images" / split / name
            label_out = out_dir / "labels" / split / f"no_eye_crop_{made:06d}.txt"
            cv2.imwrite(str(image_out), crop, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            label_out.write_text("", encoding="utf-8")
            writer.writerow([split, image_out.as_posix(), "no_eye", "", " ".join(f"{v:.1f}" for v in crop_box), sample.image_path.as_posix()])
            counts[split] += 1
            made += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", default="data/raw/WFLW_annotations/list_98pt_rect_attr_train_test")
    parser.add_argument("--images-root", default="data/raw/WFLW_images")
    parser.add_argument("--out", default="data/eye_state_yolo_wflw")
    parser.add_argument("--closed-threshold", type=float, default=0.12)
    parser.add_argument("--open-threshold", type=float, default=0.24)
    parser.add_argument("--pad-ratio", type=float, default=0.35)
    parser.add_argument("--val-ratio", type=float, default=0.12)
    parser.add_argument("--generated-negatives-per-split", type=int, default=200)
    parser.add_argument("--crop-negatives-per-split", type=int, default=500)
    parser.add_argument("--negative-size", type=int, default=320)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    samples = collect_samples(
        Path(args.annotations_root),
        Path(args.images_root),
        args.closed_threshold,
        args.open_threshold,
        args.pad_ratio,
    )
    samples = assign_validation_split(samples, args.val_ratio, args.seed)
    out_dir = Path(args.out)
    prepare_dirs(out_dir, args.clean)

    rng = random.Random(args.seed)
    meta_path = out_dir / "metadata.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as meta:
        writer = csv.writer(meta)
        writer.writerow(["split", "image", "label", "ear", "xyxy", "source"])
        counts = copy_positive_samples(samples, out_dir, writer)
        negative_counts = add_negative_samples(
            samples,
            out_dir,
            rng,
            args.generated_negatives_per_split,
            args.crop_negatives_per_split,
            args.negative_size,
            args.jpeg_quality,
            writer,
        )

    write_data_yaml(out_dir)
    print(f"[info] wrote {out_dir}")
    for split in SPLITS:
        print(
            f"[info] {split}: images={counts[split]['images']} "
            f"open={counts[split]['eye_open']} closed={counts[split]['eye_closed']} "
            f"negatives={negative_counts[split]}"
        )
    print(f"[info] metadata={meta_path}")
    print(f"[info] data_yaml={out_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
