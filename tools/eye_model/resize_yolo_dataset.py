#!/usr/bin/env python3
"""Resize a YOLO image dataset to a fixed square size without changing labels."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}


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


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def label_path_for(image_path: Path, image_root: Path, label_root: Path) -> Path:
    rel = image_path.relative_to(image_root)
    return (label_root / rel).with_suffix(".txt")


def resize_split(src_images: Path, dst_images: Path, dst_labels: Path, size: int) -> tuple[int, int]:
    src_labels = Path(str(src_images).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
    images = list_images(src_images)
    written = 0
    copied_labels = 0
    for src_image in images:
        rel = src_image.relative_to(src_images)
        dst_image = dst_images / rel
        dst_label = (dst_labels / rel).with_suffix(".txt")
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(src_image), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[warn] failed to read {src_image}", flush=True)
            continue
        if image.shape[0] != size or image.shape[1] != size:
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(dst_image), image)
        written += 1

        src_label = label_path_for(src_image, src_images, src_labels)
        if src_label.exists():
            shutil.copyfile(src_label, dst_label)
        else:
            dst_label.write_text("", encoding="utf-8")
        copied_labels += 1
    return written, copied_labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="source YOLO data.yaml")
    parser.add_argument("--out", required=True, help="output dataset directory")
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()

    data_yaml = Path(args.data)
    cfg = load_yaml(data_yaml)
    src_root = resolve_dataset_dir(data_yaml, cfg.get("path", data_yaml.parent))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        if split not in cfg:
            continue
        src_images = resolve_dataset_dir(data_yaml, Path(src_root) / cfg[split])
        dst_images = out_root / "images" / split
        dst_labels = out_root / "labels" / split
        count, labels = resize_split(src_images, dst_images, dst_labels, args.imgsz)
        print(f"[info] {split}: images={count} labels={labels}", flush=True)

    names = cfg.get("names", {0: "eye_open", 1: "eye_closed"})
    data_text = (
        f"path: {out_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
    )
    if isinstance(names, dict):
        for key in sorted(names, key=lambda item: int(item)):
            data_text += f"  {int(key)}: {names[key]}\n"
    else:
        for idx, name in enumerate(names):
            data_text += f"  {idx}: {name}\n"
    (out_root / "data.yaml").write_text(data_text, encoding="utf-8")
    print(f"[info] wrote {out_root / 'data.yaml'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
