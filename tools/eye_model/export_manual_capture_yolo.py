#!/usr/bin/env python3
"""Export manually annotated helmet capture images to a YOLO dataset.

This exporter only reads:

- `<capture-dir>/eye_open`
- `<capture-dir>/eye_closed`
- `<capture-dir>/no_eye`
- `<capture-dir>/manual_labels`

It intentionally does not read `data/eye_state_negative`.
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
LABELS = ("eye_open", "eye_closed", "no_eye")
SPLITS = ("train", "val", "test")
PROHIBITED_PART = "eye_state_negative"


@dataclass(frozen=True)
class ExportSample:
    image: Path
    label: Path
    class_name: str


def imread_color(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"empty image: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image


def collect_samples(capture_dir: Path) -> list[ExportSample]:
    samples: list[ExportSample] = []
    for class_name in LABELS:
        image_dir = capture_dir / class_name
        if not image_dir.exists():
            continue
        for image in sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            label = capture_dir / "manual_labels" / class_name / f"{image.stem}.txt"
            if class_name == "no_eye":
                samples.append(ExportSample(image=image, label=label, class_name=class_name))
                continue
            if not label.exists():
                continue
            if not label.read_text(encoding="utf-8").strip():
                continue
            samples.append(ExportSample(image=image, label=label, class_name=class_name))
    return samples


def split_samples(
    samples: list[ExportSample],
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[str, list[ExportSample]]:
    by_class = {label: [] for label in LABELS}
    for sample in samples:
        by_class[sample.class_name].append(sample)

    splits = {split: [] for split in SPLITS}
    for items in by_class.values():
        shuffled = list(items)
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


def write_data_yaml(out_dir: Path) -> None:
    (out_dir / "data.yaml").write_text(
        "\n".join(
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
        ),
        encoding="utf-8",
    )


def prepare_dirs(out_dir: Path, clean: bool) -> None:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def export(args: argparse.Namespace) -> None:
    capture_dir = Path(args.capture_dir)
    out_dir = Path(args.out)
    if PROHIBITED_PART in {part.lower() for part in capture_dir.parts}:
        raise SystemExit(f"refusing prohibited source path: {capture_dir}")
    if PROHIBITED_PART in {part.lower() for part in out_dir.parts}:
        raise SystemExit(f"refusing prohibited output path: {out_dir}")
    samples = collect_samples(capture_dir)
    if not samples:
        raise SystemExit(
            f"no manual labels found under {capture_dir / 'manual_labels'}; "
            "run manual_capture_annotator.py first"
        )

    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("invalid split ratios")

    prepare_dirs(out_dir, args.clean)
    rng = random.Random(args.seed)
    splits = split_samples(samples, args.val_ratio, args.test_ratio, rng)
    counts = {split: {"total": 0, "eye_open": 0, "eye_closed": 0, "no_eye": 0} for split in SPLITS}

    with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as meta_file:
        writer = csv.writer(meta_file)
        writer.writerow(["split", "image", "label", "source", "manual_label"])
        for split, split_samples_list in splits.items():
            for idx, sample in enumerate(split_samples_list):
                image = imread_color(sample.image)
                stem = f"{sample.class_name}_{split}_{idx:06d}"
                dst_image = out_dir / "images" / split / f"{stem}.jpg"
                dst_label = out_dir / "labels" / split / f"{stem}.txt"
                cv2.imwrite(str(dst_image), image, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                if sample.class_name == "no_eye":
                    dst_label.write_text("", encoding="utf-8")
                else:
                    shutil.copyfile(sample.label, dst_label)
                counts[split]["total"] += 1
                counts[split][sample.class_name] += 1
                writer.writerow([split, dst_image.as_posix(), sample.class_name, sample.image.as_posix(), sample.label.as_posix()])

    write_data_yaml(out_dir)
    readme = [
        "# Manual helmet capture YOLO dataset",
        "",
        f"- capture_dir: `{capture_dir.as_posix()}`",
        f"- manual_labels: `{(capture_dir / 'manual_labels').as_posix()}`",
        "- Source restriction: this exporter does not read `data/eye_state_negative`.",
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
    for split in SPLITS:
        print(f"[info] {split}: {counts[split]}")
    print(f"[info] data_yaml={out_dir / 'data.yaml'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", default="data/eye_state_capture")
    parser.add_argument("--out", default="data/eye_state_yolo_capture_manual_v1")
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
