#!/usr/bin/env python3
"""Prepare a YOLO eye-state dataset from locally captured helmet eye images.

Expected capture layout:

    data/eye_state_capture/eye_open/*.bmp
    data/eye_state_capture/eye_closed/*.bmp
    data/eye_state_capture/no_eye/*.bmp

The captured open/closed images are target-domain near-eye patches. For those
images this script writes one centered eye box. For no-eye images it writes an
empty label. A visualization folder is written under the capture directory so
the generated labels can be inspected before or after training.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard.
    raise SystemExit(
        "opencv-python and numpy are required.\n"
        f"Install them in this environment:\n  {sys.executable} -m pip install opencv-python numpy"
    ) from exc


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
CLASS_TO_ID = {"eye_open": 0, "eye_closed": 1}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class CaptureSample:
    path: Path
    label: str


def imread_color(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"empty image: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image


def collect_samples(capture_dir: Path) -> list[CaptureSample]:
    samples: list[CaptureSample] = []
    for label in ("eye_open", "eye_closed", "no_eye"):
        label_dir = capture_dir / label
        if not label_dir.exists():
            continue
        for path in sorted(p for p in label_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            samples.append(CaptureSample(path=path, label=label))
    return samples


def split_samples(
    samples: list[CaptureSample],
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[str, list[CaptureSample]]:
    by_label: dict[str, list[CaptureSample]] = {"eye_open": [], "eye_closed": [], "no_eye": []}
    for sample in samples:
        by_label[sample.label].append(sample)

    splits = {split: [] for split in SPLITS}
    for label_samples in by_label.values():
        shuffled = list(label_samples)
        rng.shuffle(shuffled)
        total = len(shuffled)
        test_count = int(total * test_ratio)
        val_count = int(total * val_ratio)
        splits["test"].extend(shuffled[:test_count])
        splits["val"].extend(shuffled[test_count:test_count + val_count])
        splits["train"].extend(shuffled[test_count + val_count:])

    for split in SPLITS:
        rng.shuffle(splits[split])
    return splits


def center_bbox(width: int, height: int, scale: float) -> tuple[float, float, float, float]:
    box_w = width * scale
    box_h = height * scale
    return width * 0.5, height * 0.5, box_w, box_h


def dark_region_bbox(
    image: np.ndarray,
    box_scale: float,
    min_width_ratio: float,
    min_height_ratio: float,
) -> tuple[float, float, float, float]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = float(np.percentile(blur, 35))
    mask = (blur <= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return center_bbox(width, height, box_scale)

    image_cx = width * 0.5
    image_cy = height * 0.42
    best_idx = 1
    best_score = float("-inf")
    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < max(12, width * height * 0.002):
            continue
        cx = x + w * 0.5
        cy = y + h * 0.5
        centrality = 1.0 - min(1.0, (((cx - image_cx) / width) ** 2 + ((cy - image_cy) / height) ** 2) ** 0.5)
        score = float(area) * (0.55 + 0.45 * centrality)
        if score > best_score:
            best_score = score
            best_idx = idx

    x, y, w, h, _area = stats[best_idx]
    cx = float(x + w * 0.5)
    cy = float(y + h * 0.5)
    box_w = max(float(w) * 2.35, width * min_width_ratio)
    box_h = max(float(h) * 3.80, height * min_height_ratio)
    box_w = min(width * box_scale, box_w)
    box_h = min(height * box_scale, box_h)

    half_w = box_w * 0.5
    half_h = box_h * 0.5
    cx = min(max(cx, half_w), width - half_w)
    cy = min(max(cy, half_h), height - half_h)
    return cx, cy, box_w, box_h


def make_bbox(
    image: np.ndarray,
    box_mode: str,
    box_scale: float,
    min_width_ratio: float,
    min_height_ratio: float,
) -> tuple[float, float, float, float]:
    height, width = image.shape[:2]
    if box_mode == "center":
        return center_bbox(width, height, box_scale)
    if box_mode == "dark":
        return dark_region_bbox(image, box_scale, min_width_ratio, min_height_ratio)
    raise ValueError(f"unknown box mode: {box_mode}")


def yolo_line(
    label: str,
    image: np.ndarray,
    box_mode: str,
    box_scale: float,
    min_width_ratio: float,
    min_height_ratio: float,
) -> str:
    height, width = image.shape[:2]
    cx, cy, box_w, box_h = make_bbox(image, box_mode, box_scale, min_width_ratio, min_height_ratio)
    return (
        f"{CLASS_TO_ID[label]} "
        f"{cx / width:.6f} {cy / height:.6f} "
        f"{box_w / width:.6f} {box_h / height:.6f}\n"
    )


def draw_visual(
    image: np.ndarray,
    label: str,
    box_mode: str,
    box_scale: float,
    min_width_ratio: float,
    min_height_ratio: float,
) -> np.ndarray:
    visual = image.copy()
    height, width = visual.shape[:2]
    if label in CLASS_TO_ID:
        cx, cy, box_w, box_h = make_bbox(image, box_mode, box_scale, min_width_ratio, min_height_ratio)
        x1 = max(0, int(round(cx - box_w / 2.0)))
        y1 = max(0, int(round(cy - box_h / 2.0)))
        x2 = min(width - 1, int(round(cx + box_w / 2.0)))
        y2 = min(height - 1, int(round(cy + box_h / 2.0)))
        color = (64, 255, 128) if label == "eye_open" else (64, 64, 255)
        cv2.rectangle(visual, (x1, y1), (x2, y2), color, 2)
    else:
        color = (255, 208, 64)
    cv2.putText(visual, f"{label} {box_mode}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return visual


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


def prepare_dirs(out_dir: Path, visual_dir: Path, clean: bool) -> None:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    if clean and visual_dir.exists():
        shutil.rmtree(visual_dir)
    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    for label in ("eye_open", "eye_closed", "no_eye"):
        (visual_dir / label).mkdir(parents=True, exist_ok=True)


def convert(args: argparse.Namespace) -> None:
    capture_dir = Path(args.capture_dir)
    out_dir = Path(args.out)
    visual_dir = capture_dir / args.visual_dir
    samples = collect_samples(capture_dir)
    if not samples:
        raise SystemExit(f"no capture images found under {capture_dir}")

    if not (0.0 < args.box_scale <= 1.0):
        raise ValueError("--box-scale must be in (0, 1]")
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("invalid split ratios")

    prepare_dirs(out_dir, visual_dir, args.clean)
    rng = random.Random(args.seed)
    by_split = split_samples(samples, args.val_ratio, args.test_ratio, rng)

    counts = {split: {"total": 0, "eye_open": 0, "eye_closed": 0, "no_eye": 0} for split in SPLITS}
    metadata_path = out_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as meta_file:
        writer = csv.writer(meta_file)
        writer.writerow(["split", "image", "label", "source", "width", "height", "box_scale"])

        for split, split_samples_list in by_split.items():
            for idx, sample in enumerate(split_samples_list):
                image = imread_color(sample.path)
                height, width = image.shape[:2]
                stem = f"{sample.label}_{split}_{idx:06d}"
                image_out = out_dir / "images" / split / f"{stem}.jpg"
                label_out = out_dir / "labels" / split / f"{stem}.txt"
                visual_out = visual_dir / sample.label / f"{stem}.jpg"

                cv2.imwrite(str(image_out), image, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                if sample.label in CLASS_TO_ID:
                    label_out.write_text(
                        yolo_line(
                            sample.label,
                            image,
                            args.box_mode,
                            args.box_scale,
                            args.min_width_ratio,
                            args.min_height_ratio,
                        ),
                        encoding="utf-8",
                    )
                else:
                    label_out.write_text("", encoding="utf-8")

                visual = draw_visual(
                    image,
                    sample.label,
                    args.box_mode,
                    args.box_scale,
                    args.min_width_ratio,
                    args.min_height_ratio,
                )
                cv2.imwrite(str(visual_out), visual, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])

                counts[split]["total"] += 1
                counts[split][sample.label] += 1
                writer.writerow([split, image_out.as_posix(), sample.label, sample.path.as_posix(), width, height, args.box_scale])

    write_data_yaml(out_dir)
    readme = [
        "# Helmet capture eye-state YOLO dataset",
        "",
        f"- capture_dir: `{capture_dir.as_posix()}`",
        f"- visual_dir: `{visual_dir.as_posix()}`",
        f"- box_mode: `{args.box_mode}`",
        f"- box_scale: `{args.box_scale}`",
        "",
        "User explicitly authorized these captured images for annotation and training.",
        "",
        "## Counts",
        "",
    ]
    for split in SPLITS:
        item = counts[split]
        readme.append(
            f"- {split}: total={item['total']} eye_open={item['eye_open']} "
            f"eye_closed={item['eye_closed']} no_eye={item['no_eye']}"
        )
    readme.append("")
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"[info] wrote dataset={out_dir}")
    print(f"[info] wrote visualizations={visual_dir}")
    for split in SPLITS:
        print(f"[info] {split}: {counts[split]}")
    print(f"[info] data_yaml={out_dir / 'data.yaml'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", default="data/eye_state_capture")
    parser.add_argument("--out", default="data/eye_state_yolo_capture_v1")
    parser.add_argument("--visual-dir", default="visualized")
    parser.add_argument("--box-mode", choices=("center", "dark"), default="center")
    parser.add_argument("--box-scale", type=float, default=0.92)
    parser.add_argument("--min-width-ratio", type=float, default=0.56)
    parser.add_argument("--min-height-ratio", type=float, default=0.34)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    convert(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
