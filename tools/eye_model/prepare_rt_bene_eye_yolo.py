#!/usr/bin/env python3
"""Prepare a YOLO eye-state dataset from RT-BENE eye patches.

RT-BENE labels use:

    0.0 -> open eyes
    1.0 -> blink / closed eyes
    0.5 -> annotator disagreement, discarded

The archive contains extracted eye image patches, so each positive sample is
written with one full-patch eye box. This is an offline dataset conversion only;
the board runtime must remain a single direct detector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_TO_ID = {"eye_open": 0, "eye_closed": 1}
ID_TO_CLASS = {0: "eye_open", 1: "eye_closed"}


@dataclass(frozen=True)
class SubjectInfo:
    subject: str
    labels_csv: Path
    left_dir: Path
    right_dir: Path
    category: str
    fold_id: int


@dataclass(frozen=True)
class Sample:
    image: Path
    class_id: int
    subject: str
    side: str
    frame_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rt-bene-root", default="data/raw/rt_bene")
    parser.add_argument("--out", default="data/eye_state_yolo_rt_bene_v1")
    parser.add_argument("--extract", action="store_true", help="extract downloaded *_noglasses_eyes.tar files")
    parser.add_argument("--clean", action="store_true", help="delete output directory first")
    parser.add_argument("--split-mode", choices=("official-fold", "image"), default="official-fold")
    parser.add_argument("--test-fold", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.15, help="only used with --split-mode image")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="only used with --split-mode image")
    parser.add_argument("--max-open-per-closed", type=float, default=6.0)
    parser.add_argument("--box-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_subjects(root: Path) -> list[SubjectInfo]:
    subjects_csv = root / "rt_bene_subjects.csv"
    if not subjects_csv.exists():
        raise FileNotFoundError(f"missing {subjects_csv}")

    subjects: list[SubjectInfo] = []
    extracted_root = root / "extracted"
    with subjects_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            subject = Path(row[1]).name.split("_")[0]
            subjects.append(
                SubjectInfo(
                    subject=subject,
                    labels_csv=root / row[1],
                    left_dir=extracted_root / row[2].strip("/"),
                    right_dir=extracted_root / row[3].strip("/"),
                    category=row[4],
                    fold_id=int(row[5]),
                )
            )
    return subjects


def extract_downloaded_archives(root: Path, subjects: list[SubjectInfo]) -> None:
    extracted_root = root / "extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        archive = root / f"{subject.subject}_noglasses_eyes.tar"
        marker = extracted_root / f".{subject.subject}_done"
        if marker.exists():
            continue
        if not archive.exists():
            continue
        print(f"[info] extracting {archive}")
        subprocess.run(["tar", "-xf", str(archive), "-C", str(extracted_root)], check=True)
        marker.write_text("ok\n", encoding="utf-8")


def load_label_rows(labels_csv: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            name = row[0].strip()
            value = row[1].strip()
            if value == "0.0":
                labels[name] = CLASS_TO_ID["eye_open"]
            elif value == "1.0":
                labels[name] = CLASS_TO_ID["eye_closed"]
    return labels


def collect_samples(subjects: list[SubjectInfo]) -> list[Sample]:
    samples: list[Sample] = []
    missing_subjects = []
    for subject in subjects:
        if subject.category == "discarded":
            continue
        if not subject.labels_csv.exists():
            continue
        if not subject.left_dir.exists() and not subject.right_dir.exists():
            missing_subjects.append(subject.subject)
            continue
        label_rows = load_label_rows(subject.labels_csv)
        for left_name, class_id in label_rows.items():
            left_image = subject.left_dir / left_name
            if left_image.exists() and left_image.suffix.lower() in IMAGE_EXTS:
                samples.append(Sample(left_image, class_id, subject.subject, "left", left_name))
            right_name = left_name.replace("left_", "right_", 1)
            right_image = subject.right_dir / right_name
            if right_image.exists() and right_image.suffix.lower() in IMAGE_EXTS:
                samples.append(Sample(right_image, class_id, subject.subject, "right", right_name))

    if missing_subjects:
        print(f"[warn] archives not extracted/downloaded for: {', '.join(sorted(missing_subjects))}")
    return samples


def cap_open_samples(samples: list[Sample], max_open_per_closed: float, seed: int) -> list[Sample]:
    by_class: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_class[sample.class_id].append(sample)

    closed = by_class[CLASS_TO_ID["eye_closed"]]
    open_samples = by_class[CLASS_TO_ID["eye_open"]]
    if not closed or max_open_per_closed <= 0:
        return samples

    max_open = int(round(len(closed) * max_open_per_closed))
    if len(open_samples) <= max_open:
        return samples

    rng = random.Random(seed)
    kept_open = open_samples[:]
    rng.shuffle(kept_open)
    kept = kept_open[:max_open] + closed
    kept.sort(key=lambda item: (item.subject, item.side, item.frame_name))
    print(f"[info] capped open samples {len(open_samples)} -> {len(kept_open[:max_open])}")
    return kept


def stable_unit(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / float(1 << 64)


def split_sample(sample: Sample, subjects: dict[str, SubjectInfo], args: argparse.Namespace) -> str:
    if args.split_mode == "official-fold":
        subject = subjects[sample.subject]
        if subject.category == "validation":
            return "val"
        if subject.fold_id == args.test_fold:
            return "test"
        return "train"

    value = stable_unit(f"{sample.subject}/{sample.side}/{sample.frame_name}")
    if value < args.test_ratio:
        return "test"
    if value < args.test_ratio + args.val_ratio:
        return "val"
    return "train"


def copy_sample(sample: Sample, split: str, out_dir: Path, box_scale: float) -> None:
    stem = f"{sample.subject}_{sample.side}_{sample.image.stem}"
    image_out = out_dir / "images" / split / f"{stem}{sample.image.suffix.lower()}"
    label_out = out_dir / "labels" / split / f"{stem}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.image, image_out)

    scale = max(0.05, min(1.0, box_scale))
    label_out.write_text(f"{sample.class_id} 0.500000 0.500000 {scale:.6f} {scale:.6f}\n", encoding="utf-8")


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


def write_readme(out_dir: Path, counts: dict[str, dict[int, int]], args: argparse.Namespace) -> None:
    lines = [
        "# RT-BENE YOLO eye-state dataset",
        "",
        "Source: https://zenodo.org/records/3685316",
        "",
        "Labels:",
        "",
        "- `0.0` -> `eye_open`",
        "- `1.0` -> `eye_closed`",
        "- `0.5` discarded",
        "",
        "The RT-BENE files are extracted eye patches. Each converted sample uses",
        "one full-patch detection box; this is an offline label conversion, not a",
        "runtime ROI/landmark/cascade path.",
        "",
        f"split_mode: `{args.split_mode}`",
        f"box_scale: `{args.box_scale}`",
        "",
        "Counts:",
        "",
    ]
    for split in ("train", "val", "test"):
        split_counts = counts.get(split, {})
        total = sum(split_counts.values())
        lines.append(
            f"- {split}: total={total} "
            f"eye_open={split_counts.get(0, 0)} eye_closed={split_counts.get(1, 0)}"
        )
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.rt_bene_root)
    out_dir = Path(args.out)

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    subjects = read_subjects(root)
    if args.extract:
        extract_downloaded_archives(root, subjects)

    samples = collect_samples(subjects)
    if not samples:
        raise SystemExit("no samples found; download and/or extract RT-BENE archives first")
    samples = cap_open_samples(samples, args.max_open_per_closed, args.seed)

    by_subject = {subject.subject: subject for subject in subjects}
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        split = split_sample(sample, by_subject, args)
        copy_sample(sample, split, out_dir, args.box_scale)
        counts[split][sample.class_id] += 1

    write_data_yaml(out_dir)
    write_readme(out_dir, counts, args)

    total_open = sum(split_counts.get(0, 0) for split_counts in counts.values())
    total_closed = sum(split_counts.get(1, 0) for split_counts in counts.values())
    print(f"[info] wrote {out_dir}")
    print(f"[info] total eye_open={total_open} eye_closed={total_closed}")
    for split in ("train", "val", "test"):
        split_counts = counts.get(split, {})
        print(
            f"[info] {split}: eye_open={split_counts.get(0, 0)} "
            f"eye_closed={split_counts.get(1, 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
