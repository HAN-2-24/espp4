#!/usr/bin/env python3
"""Prepare a YOLO eye-state dataset from MRL Eye Dataset files.

MRL eye image filenames encode eye state. This script parses those labels and
then reuses the synthetic full-frame generator:

    eye_state 0 -> eye_closed
    eye_state 1 -> eye_open

The generated detector dataset is a bootstrap dataset for pipeline integration,
not final helmet-camera validation data.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import shutil
import ssl
import sys
import zipfile
from pathlib import Path

from prepare_synthetic_eye_yolo import SourceImage, build_dataset_from_sources

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - dependency guard.
    requests = None


DEFAULT_MRL_URLS = [
    "https://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip",
    "http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_mrl_label(path: Path) -> str | None:
    parts = path.stem.split("_")
    if len(parts) < 5:
        return None
    eye_state = parts[4]
    if eye_state == "1":
        return "eye_open"
    if eye_state == "0":
        return "eye_closed"
    return None


def download_single(url: str, out_path: Path, insecure: bool) -> None:
    import urllib.request

    context = ssl._create_unverified_context() if insecure else None
    with urllib.request.urlopen(url, timeout=30, context=context) as response, out_path.open("wb") as f:
        shutil.copyfileobj(response, f, length=1024 * 1024)


def download_range(url: str, part_path: Path, start: int, end: int, insecure: bool) -> None:
    if requests is None:
        raise RuntimeError("requests is required for segmented download")
    expected = end - start + 1
    existing = part_path.stat().st_size if part_path.exists() else 0
    if existing == expected:
        return
    if existing > expected:
        part_path.unlink()
        existing = 0

    range_start = start + existing
    headers = {"Range": f"bytes={range_start}-{end}"}
    with requests.get(url, headers=headers, stream=True, timeout=60, verify=not insecure) as response:
        if response.status_code not in (200, 206):
            raise RuntimeError(f"range {range_start}-{end} failed: HTTP {response.status_code}")
        if existing > 0 and response.status_code != 206:
            raise RuntimeError(f"range resume {range_start}-{end} did not return HTTP 206")
        mode = "ab" if existing else "wb"
        with part_path.open(mode) as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    actual = part_path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"range {start}-{end} size mismatch: got {actual}, expected {expected}")


def download_segmented(url: str, out_path: Path, insecure: bool, segments: int) -> None:
    if requests is None:
        raise RuntimeError("requests is required for segmented download")
    head = requests.head(url, timeout=30, verify=not insecure)
    head.raise_for_status()
    size = int(head.headers.get("Content-Length", "0"))
    accept_ranges = head.headers.get("Accept-Ranges", "").lower()
    if size <= 0 or "bytes" not in accept_ranges:
        print("[warn] server does not advertise byte ranges; falling back to single download")
        download_single(url, out_path, insecure)
        return

    tmp_dir = out_path.with_suffix(out_path.suffix + ".parts")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = (size + segments - 1) // segments
    jobs = []
    for idx in range(segments):
        start = idx * chunk_size
        if start >= size:
            break
        end = min(size - 1, start + chunk_size - 1)
        jobs.append((idx, start, end, tmp_dir / f"part_{idx:02d}.bin"))

    print(f"[info] segmented download size={size} segments={len(jobs)}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [
            executor.submit(download_range, url, part_path, start, end, insecure)
            for _idx, start, end, part_path in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            done = sum(1 for item in jobs if item[3].exists() and item[3].stat().st_size == item[2] - item[1] + 1)
            print(f"[info] downloaded segment {done}/{len(jobs)}")

    with out_path.open("wb") as out_file:
        for _idx, _start, _end, part_path in jobs:
            with part_path.open("rb") as part_file:
                shutil.copyfileobj(part_file, out_file, length=1024 * 1024)

    actual = out_path.stat().st_size
    if actual != size:
        raise RuntimeError(f"combined zip size mismatch: got {actual}, expected {size}")
    shutil.rmtree(tmp_dir)


def download_file(urls: list[str], out_path: Path, insecure: bool, segments: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in urls:
        try:
            print(f"[info] downloading {url}")
            if segments > 1:
                download_segmented(url, out_path, insecure, segments)
            else:
                download_single(url, out_path, insecure)
            print(f"[info] downloaded {out_path} {out_path.stat().st_size} bytes")
            return
        except Exception as exc:  # pragma: no cover - network dependent.
            last_error = exc
            print(f"[warn] download failed: {url}: {exc}")
    raise RuntimeError(f"all download URLs failed; last error: {last_error}")


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".mrl_extract_done"
    if marker.exists():
        print(f"[info] using existing extracted dataset {extract_dir}")
        return extract_dir
    print(f"[info] extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return extract_dir


def collect_mrl_sources(root: Path, max_per_class: int | None) -> list[SourceImage]:
    if not root.exists():
        raise FileNotFoundError(root)

    by_label = {"eye_open": [], "eye_closed": []}
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        label = parse_mrl_label(path)
        if label in by_label:
            by_label[label].append(SourceImage(path=path, label=label))

    if max_per_class is not None:
        by_label = {label: items[:max_per_class] for label, items in by_label.items()}

    print(f"[info] MRL open={len(by_label['eye_open'])} closed={len(by_label['eye_closed'])}")
    if not by_label["eye_open"] or not by_label["eye_closed"]:
        raise ValueError("MRL labels not found; expected filenames with eye state as the 5th underscore-separated field")

    return by_label["eye_open"] + by_label["eye_closed"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrl-root", help="existing extracted MRL dataset root")
    parser.add_argument("--download", action="store_true", help="download MRL zip before preparing dataset")
    parser.add_argument("--insecure", action="store_true", help="allow HTTPS download with an unverified certificate")
    parser.add_argument("--segments", type=int, default=8, help="parallel HTTP range segments for download")
    parser.add_argument("--url", action="append", help="override/add MRL download URL")
    parser.add_argument("--zip", default="data/raw/mrlEyes_2018_01.zip", help="MRL zip path")
    parser.add_argument("--extract-dir", default="data/raw/mrlEyes_2018_01", help="MRL extraction directory")
    parser.add_argument("--out", default="data/eye_state_yolo_mrl", help="output YOLO dataset directory")
    parser.add_argument("--background-dir", help="optional background images for synthetic full-frame canvases")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--samples-per-image", type=int, default=2)
    parser.add_argument("--min-eye-width-ratio", type=float, default=0.28)
    parser.add_argument("--max-eye-width-ratio", type=float, default=0.72)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--max-per-class", type=int)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="delete output directory first")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if args.download and not zip_path.exists():
        urls = args.url or DEFAULT_MRL_URLS
        download_file(urls, zip_path, args.insecure, max(1, args.segments))

    if args.mrl_root:
        mrl_root = Path(args.mrl_root)
    elif zip_path.exists():
        mrl_root = extract_zip(zip_path, Path(args.extract_dir))
    else:
        raise SystemExit("provide --mrl-root or use --download")

    sources = collect_mrl_sources(mrl_root, args.max_per_class)
    build_dataset_from_sources(sources, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
