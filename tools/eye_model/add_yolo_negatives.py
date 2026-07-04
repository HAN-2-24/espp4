#!/usr/bin/env python3
"""Add no-object negative images to a YOLO detection dataset.

YOLO detection does not need a `no_eye` class. It does need background images
with empty label files so objectness learns not to fire on every frame.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "opencv-python and numpy are required.\n"
        f"Install them in this environment:\n  {sys.executable} -m pip install opencv-python numpy"
    ) from exc


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "test")


def list_images(paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for root in paths:
        if not root.exists():
            raise FileNotFoundError(root)
        if root.is_file() and root.suffix.lower() in IMAGE_EXTS:
            images.append(root)
        elif root.is_dir():
            images.extend(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return sorted(images)


def copy_dataset(src: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"output dataset already exists: {out}")
    if not (src / "data.yaml").exists():
        raise FileNotFoundError(src / "data.yaml")
    shutil.copytree(src, out)
    rewrite_data_yaml_path(out / "data.yaml", out)


def rewrite_data_yaml_path(data_yaml: Path, dataset_root: Path) -> None:
    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    replaced = False
    updated = []
    for line in lines:
        if line.strip().startswith("path:") and not replaced:
            updated.append(f"path: {dataset_root.as_posix()}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.insert(0, f"path: {dataset_root.as_posix()}")
    data_yaml.write_text("\n".join(updated) + "\n", encoding="utf-8")


def write_image(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def random_solid(rng: random.Random, imgsz: int) -> np.ndarray:
    value = rng.randint(0, 255)
    return np.full((imgsz, imgsz, 3), value, dtype=np.uint8)


def random_gradient(rng: random.Random, imgsz: int) -> np.ndarray:
    a = rng.randint(0, 255)
    b = rng.randint(0, 255)
    line = np.linspace(a, b, imgsz, dtype=np.float32)
    if rng.random() < 0.5:
        plane = np.tile(line, (imgsz, 1))
    else:
        plane = np.tile(line[:, None], (1, imgsz))
    image = np.dstack([
        np.clip(plane + rng.randint(-25, 25), 0, 255),
        np.clip(plane + rng.randint(-25, 25), 0, 255),
        np.clip(plane + rng.randint(-25, 25), 0, 255),
    ]).astype(np.uint8)
    return image


def random_noise(rng: random.Random, imgsz: int) -> np.ndarray:
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    low = rng.randint(0, 160)
    high = rng.randint(max(low + 1, 32), 256)
    image = np_rng.integers(low, high, (imgsz, imgsz, 3), dtype=np.uint8)
    if rng.random() < 0.45:
        image = cv2.GaussianBlur(image, (rng.choice([3, 5, 7]), rng.choice([3, 5, 7])), 0)
    return image


def random_checker(rng: random.Random, imgsz: int) -> np.ndarray:
    block = rng.randint(4, 40)
    yx = np.indices((imgsz, imgsz)).sum(axis=0)
    checker = ((yx // block) % 2 * rng.randint(80, 255)).astype(np.uint8)
    image = cv2.merge((checker, checker, checker))
    tint = np.array([rng.randint(-30, 30), rng.randint(-30, 30), rng.randint(-30, 30)], dtype=np.int16)
    return np.clip(image.astype(np.int16) + tint, 0, 255).astype(np.uint8)


def random_rectangles(rng: random.Random, imgsz: int) -> np.ndarray:
    image = np.full((imgsz, imgsz, 3), rng.randint(20, 235), dtype=np.uint8)
    for _ in range(rng.randint(2, 12)):
        color = tuple(rng.randint(0, 255) for _ in range(3))
        x1 = rng.randint(0, imgsz - 1)
        y1 = rng.randint(0, imgsz - 1)
        x2 = rng.randint(x1, imgsz - 1)
        y2 = rng.randint(y1, imgsz - 1)
        thickness = -1 if rng.random() < 0.7 else rng.randint(1, 4)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if rng.random() < 0.4:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    return image


def random_lines_text(rng: random.Random, imgsz: int) -> np.ndarray:
    image = np.full((imgsz, imgsz, 3), rng.randint(40, 230), dtype=np.uint8)
    for _ in range(rng.randint(4, 20)):
        color = tuple(rng.randint(0, 255) for _ in range(3))
        p1 = (rng.randint(0, imgsz - 1), rng.randint(0, imgsz - 1))
        p2 = (rng.randint(0, imgsz - 1), rng.randint(0, imgsz - 1))
        cv2.line(image, p1, p2, color, rng.randint(1, 4))
    if rng.random() < 0.6:
        text = rng.choice(["NO EYE", "TEST", "123", "BG", "HELMET"])
        cv2.putText(
            image,
            text,
            (rng.randint(0, imgsz // 3), rng.randint(imgsz // 4, imgsz - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            rng.uniform(0.5, 1.6),
            tuple(rng.randint(0, 255) for _ in range(3)),
            rng.randint(1, 3),
            cv2.LINE_AA,
        )
    return image


def generate_negative(rng: random.Random, imgsz: int) -> np.ndarray:
    maker = rng.choice([
        random_solid,
        random_gradient,
        random_noise,
        random_checker,
        random_rectangles,
        random_lines_text,
    ])
    return maker(rng, imgsz)


def split_counts(total: int, val_ratio: float, test_ratio: float) -> dict[str, int]:
    test = int(total * test_ratio)
    val = int(total * val_ratio)
    train = total - val - test
    return {"train": train, "val": val, "test": test}


def copy_external_negatives(
    out: Path,
    images: list[Path],
    rng: random.Random,
    imgsz: int,
    jpeg_quality: int,
    counts: dict[str, int],
    writer: csv.writer,
) -> dict[str, int]:
    shuffled = list(images)
    rng.shuffle(shuffled)
    index = 0
    written = {split: 0 for split in SPLITS}
    for split in SPLITS:
        for _ in range(counts[split]):
            if not shuffled:
                return written
            src = shuffled[index % len(shuffled)]
            index += 1
            image = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            stem = f"no_eye_external_{written[split]:06d}"
            image_path = out / "images" / split / f"{stem}.jpg"
            label_path = out / "labels" / split / f"{stem}.txt"
            write_image(image_path, image, jpeg_quality)
            label_path.write_text("", encoding="utf-8")
            writer.writerow([split, image_path.as_posix(), "no_eye", src.as_posix(), "external"])
            written[split] += 1
    return written


def add_generated_negatives(
    out: Path,
    rng: random.Random,
    imgsz: int,
    jpeg_quality: int,
    counts: dict[str, int],
    writer: csv.writer,
) -> dict[str, int]:
    written = {split: 0 for split in SPLITS}
    for split in SPLITS:
        for idx in range(counts[split]):
            image = generate_negative(rng, imgsz)
            stem = f"no_eye_generated_{idx:06d}"
            image_path = out / "images" / split / f"{stem}.jpg"
            label_path = out / "labels" / split / f"{stem}.txt"
            write_image(image_path, image, jpeg_quality)
            label_path.write_text("", encoding="utf-8")
            writer.writerow([split, image_path.as_posix(), "no_eye", "", "generated"])
            written[split] += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="source YOLO dataset directory")
    parser.add_argument("--out", required=True, help="new YOLO dataset directory")
    parser.add_argument("--negative-dir", action="append", default=[], help="real no-eye image directory/file")
    parser.add_argument("--generated", type=int, default=0, help="number of generated no-eye images")
    parser.add_argument("--external", type=int, default=0, help="number of external no-eye images to copy")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.generated <= 0 and args.external <= 0:
        raise SystemExit("nothing to add: pass --generated and/or --external with --negative-dir")
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("invalid split ratios")

    src = Path(args.src)
    out = Path(args.out)
    copy_dataset(src, out)
    rng = random.Random(args.seed)

    meta_path = out / "negative_metadata.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as meta_file:
        writer = csv.writer(meta_file)
        writer.writerow(["split", "image", "label", "source", "kind"])

        generated_counts = split_counts(args.generated, args.val_ratio, args.test_ratio)
        generated_written = add_generated_negatives(out, rng, args.imgsz, args.jpeg_quality, generated_counts, writer)

        external_written = {split: 0 for split in SPLITS}
        if args.external > 0:
            external_images = list_images([Path(p) for p in args.negative_dir])
            external_counts = split_counts(args.external, args.val_ratio, args.test_ratio)
            external_written = copy_external_negatives(
                out,
                external_images,
                rng,
                args.imgsz,
                args.jpeg_quality,
                external_counts,
                writer,
            )

    print(f"[info] copied {src} -> {out}")
    print(f"[info] generated_negatives={generated_written}")
    print(f"[info] external_negatives={external_written}")
    print(f"[info] metadata={meta_path}")
    print(f"[info] data_yaml={out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
