#!/usr/bin/env python3
"""Merge multiple YOLO eye-state datasets into one dataset."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "test")


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
    parser.add_argument("--data", action="append", required=True, help="source data.yaml; repeatable")
    parser.add_argument("--out", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {split: 0 for split in SPLITS}
    for dataset_idx, raw_yaml in enumerate(args.data):
        data_yaml = Path(raw_yaml)
        cfg = load_yaml(data_yaml)
        root = resolve_path(data_yaml, cfg.get("path", data_yaml.parent))
        prefix = f"d{dataset_idx:02d}"
        for split in SPLITS:
            if split not in cfg:
                continue
            image_root = resolve_path(data_yaml, Path(root) / cfg[split])
            label_root = Path(str(image_root).replace("\\images\\", "\\labels\\"))
            for image_idx, src_image in enumerate(list_images(image_root)):
                rel = src_image.relative_to(image_root)
                src_label = (label_root / rel).with_suffix(".txt")
                stem = f"{prefix}_{split}_{image_idx:06d}"
                dst_image = out / "images" / split / f"{stem}{src_image.suffix.lower()}"
                dst_label = out / "labels" / split / f"{stem}.txt"
                shutil.copyfile(src_image, dst_image)
                if src_label.exists():
                    shutil.copyfile(src_label, dst_label)
                else:
                    dst_label.write_text("", encoding="utf-8")
                counts[split] += 1

    write_data_yaml(out)
    print(f"[info] wrote {out}", flush=True)
    for split in SPLITS:
        print(f"[info] {split}: images={counts[split]}", flush=True)
    print(f"[info] data_yaml={out / 'data.yaml'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
