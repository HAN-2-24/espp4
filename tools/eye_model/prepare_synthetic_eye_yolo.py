#!/usr/bin/env python3
"""Build a YOLO eye-state detection dataset from cropped open/closed eye images.

This is a bootstrap path for times when helmet-view samples are not available
yet. It pastes cropped eye images into synthetic full-frame canvases and writes
YOLO labels:

    0 eye_open
    1 eye_closed

The generated dataset is useful for proving the detector/training/export/ESP-DL
pipeline. It is not a substitute for real helmet camera data.
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


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = {"eye_open": 0, "eye_closed": 1}


@dataclass(frozen=True)
class SourceImage:
    path: Path
    label: str


def collect_images(paths: list[Path], label: str, max_count: int | None) -> list[SourceImage]:
    images: list[SourceImage] = []
    for root in paths:
        if not root.exists():
            raise FileNotFoundError(root)
        if root.is_file():
            candidates = [root]
        else:
            candidates = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        for path in candidates:
            if path.suffix.lower() in IMAGE_EXTS:
                images.append(SourceImage(path=path, label=label))
    if max_count is not None:
        images = images[:max_count]
    return images


def split_sources(
    sources: list[SourceImage],
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[str, list[SourceImage]]:
    shuffled = list(sources)
    rng.shuffle(shuffled)
    total = len(shuffled)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)
    return {
        "test": shuffled[:test_count],
        "val": shuffled[test_count:test_count + val_count],
        "train": shuffled[test_count + val_count:],
    }


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image


def make_background(imgsz: int, rng: random.Random, backgrounds: list[Path]) -> np.ndarray:
    if backgrounds:
        bg_path = rng.choice(backgrounds)
        bg = read_image(bg_path)
        h, w = bg.shape[:2]
        if h >= imgsz and w >= imgsz:
            x = rng.randint(0, w - imgsz)
            y = rng.randint(0, h - imgsz)
            bg = bg[y:y + imgsz, x:x + imgsz]
        else:
            bg = cv2.resize(bg, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        return bg

    base = rng.randint(18, 115)
    canvas = np.full((imgsz, imgsz, 3), base, dtype=np.uint8)
    noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, rng.uniform(3.0, 18.0), canvas.shape)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    tint = np.array([rng.randint(-12, 12), rng.randint(-12, 12), rng.randint(-12, 12)], dtype=np.int16)
    return np.clip(canvas.astype(np.int16) + tint, 0, 255).astype(np.uint8)


def augment_eye(crop: np.ndarray, rng: random.Random) -> np.ndarray:
    eye = crop.copy()
    alpha = rng.uniform(0.65, 1.35)
    beta = rng.randint(-30, 30)
    eye = cv2.convertScaleAbs(eye, alpha=alpha, beta=beta)
    if rng.random() < 0.25:
        k = rng.choice([3, 5])
        eye = cv2.GaussianBlur(eye, (k, k), 0)
    if rng.random() < 0.35:
        eye = cv2.flip(eye, 1)
    return eye


def paste_eye(
    canvas: np.ndarray,
    eye: np.ndarray,
    imgsz: int,
    rng: random.Random,
    min_width_ratio: float,
    max_width_ratio: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = eye.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("empty eye crop")

    target_w = int(imgsz * rng.uniform(min_width_ratio, max_width_ratio))
    target_w = max(12, min(target_w, imgsz - 4))
    target_h = max(8, int(target_w * h / w))
    if target_h > imgsz - 4:
        target_h = imgsz - 4
        target_w = max(12, int(target_h * w / h))

    eye = cv2.resize(eye, (target_w, target_h), interpolation=cv2.INTER_AREA)

    max_x = max(0, imgsz - target_w)
    max_y = max(0, imgsz - target_h)
    x = rng.randint(0, max_x)
    y = rng.randint(0, max_y)

    canvas[y:y + target_h, x:x + target_w] = eye
    return canvas, (x, y, target_w, target_h)


def yolo_line(label: str, bbox: tuple[int, int, int, int], imgsz: int) -> str:
    x, y, w, h = bbox
    xc = (x + w / 2.0) / imgsz
    yc = (y + h / 2.0) / imgsz
    bw = w / imgsz
    bh = h / imgsz
    return f"{CLASSES[label]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def write_data_yaml(out_dir: Path) -> None:
    text = "\n".join([
        f"path: {out_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        "  0: eye_open",
        "  1: eye_closed",
        "",
    ])
    (out_dir / "data.yaml").write_text(text, encoding="utf-8")


def prepare_dirs(out_dir: Path, clean: bool) -> None:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def background_paths(path: Path | None) -> list[Path]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def build_dataset_from_sources(sources: list[SourceImage], args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    prepare_dirs(out_dir, args.clean)

    if not sources:
        raise ValueError("no source images found")

    by_split = split_sources(sources, args.val_ratio, args.test_ratio, rng)
    backgrounds = background_paths(Path(args.background_dir) if args.background_dir else None)
    meta_path = out_dir / "metadata.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as meta_file:
        writer = csv.writer(meta_file)
        writer.writerow(["split", "image", "label", "source", "x", "y", "w", "h"])

        for split, split_sources_list in by_split.items():
            for src_index, src in enumerate(split_sources_list):
                crop = read_image(src.path)
                for sample_idx in range(args.samples_per_image):
                    canvas = make_background(args.imgsz, rng, backgrounds)
                    eye = augment_eye(crop, rng)
                    canvas, bbox = paste_eye(
                        canvas,
                        eye,
                        args.imgsz,
                        rng,
                        args.min_eye_width_ratio,
                        args.max_eye_width_ratio,
                    )
                    stem = f"{src.label}_{src_index:06d}_{sample_idx:02d}"
                    image_path = out_dir / "images" / split / f"{stem}.jpg"
                    label_path = out_dir / "labels" / split / f"{stem}.txt"
                    cv2.imwrite(str(image_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                    label_path.write_text(yolo_line(src.label, bbox, args.imgsz), encoding="utf-8")
                    writer.writerow([split, image_path.as_posix(), src.label, src.path.as_posix(), *bbox])

    write_data_yaml(out_dir)
    counts = {split: len(items) * args.samples_per_image for split, items in by_split.items()}
    print(f"[info] wrote {out_dir}")
    print(f"[info] images train={counts['train']} val={counts['val']} test={counts['test']}")
    print(f"[info] data_yaml={out_dir / 'data.yaml'}")


def build_dataset(args: argparse.Namespace) -> None:
    sources = []
    sources.extend(collect_images([Path(p) for p in args.open_dir], "eye_open", args.max_per_class))
    sources.extend(collect_images([Path(p) for p in args.closed_dir], "eye_closed", args.max_per_class))
    build_dataset_from_sources(sources, args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-dir", nargs="+", required=True, help="directories/files of cropped open-eye images")
    parser.add_argument("--closed-dir", nargs="+", required=True, help="directories/files of cropped closed-eye images")
    parser.add_argument("--out", default="data/eye_state_yolo", help="output YOLO dataset directory")
    parser.add_argument("--background-dir", help="optional background images for synthetic full-frame canvases")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--samples-per-image", type=int, default=3)
    parser.add_argument("--min-eye-width-ratio", type=float, default=0.28)
    parser.add_argument("--max-eye-width-ratio", type=float, default=0.72)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--max-per-class", type=int)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="delete output directory first")
    args = parser.parse_args()

    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("invalid split ratios")
    if args.min_eye_width_ratio <= 0 or args.max_eye_width_ratio > 1:
        raise ValueError("eye width ratios must be in (0, 1]")
    if args.min_eye_width_ratio > args.max_eye_width_ratio:
        raise ValueError("min eye width ratio must be <= max")

    build_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
