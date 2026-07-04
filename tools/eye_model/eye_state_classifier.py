#!/usr/bin/env python3
"""Train and inspect an ESP-DL-friendly eye-state classifier.

Contract:

    input:  1x3x160x160 RGB image by default
    output: logits [1,3]

Class order:

    0 eye_open
    1 eye_closed
    2 background

The training path refuses to read data/eye_state_negative.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
CLASS_NAMES = ("eye_open", "eye_closed", "background")
PROHIBITED_PART = "eye_state_negative"


@dataclass(frozen=True)
class Sample:
    path: Path
    class_id: int


def is_prohibited_path(path: Path) -> bool:
    return PROHIBITED_PART in str(path).replace("\\", "/").lower().split("/")


def require_allowed_training_path(path: Path) -> None:
    if is_prohibited_path(path):
        raise SystemExit(f"refusing to train from prohibited path: {path}")


def imread_color(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"empty image: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def collect_class_samples(paths: list[Path], class_id: int, max_count: int, seed: int) -> list[Sample]:
    images: list[Path] = []
    for path in paths:
        require_allowed_training_path(path)
        images.extend(list_images(path))
    images = sorted(set(images))
    if max_count > 0 and len(images) > max_count:
        rng = random.Random(seed + class_id * 1009)
        images = rng.sample(images, max_count)
        images.sort()
    return [Sample(path=path, class_id=class_id) for path in images]


def split_samples(
    samples: list[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    by_class: dict[int, list[Sample]] = {idx: [] for idx in range(len(CLASS_NAMES))}
    for sample in samples:
        by_class[sample.class_id].append(sample)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for class_id, class_samples in by_class.items():
        class_samples = list(class_samples)
        rng.shuffle(class_samples)
        total = len(class_samples)
        test_count = int(round(total * test_ratio))
        val_count = int(round(total * val_ratio))
        test_count = min(test_count, max(0, total - 1)) if total > 1 else 0
        val_count = min(val_count, max(0, total - test_count - 1)) if total - test_count > 1 else 0
        splits["test"].extend(class_samples[:test_count])
        splits["val"].extend(class_samples[test_count:test_count + val_count])
        splits["train"].extend(class_samples[test_count + val_count:])

    for split_samples_ in splits.values():
        rng.shuffle(split_samples_)
    return splits


def class_counts(samples: list[Sample]) -> list[int]:
    counts = [0 for _ in CLASS_NAMES]
    for sample in samples:
        counts[sample.class_id] += 1
    return counts


def apply_train_aug(image: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        image = cv2.flip(image, 1)

    height, width = image.shape[:2]
    center = (width * 0.5, height * 0.5)
    angle = rng.uniform(-5.0, 5.0)
    scale = rng.uniform(0.88, 1.12)
    tx = rng.uniform(-0.06, 0.06) * width
    ty = rng.uniform(-0.06, 0.06) * height
    mat = cv2.getRotationMatrix2D(center, angle, scale)
    mat[0, 2] += tx
    mat[1, 2] += ty
    image = cv2.warpAffine(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    alpha = rng.uniform(0.72, 1.32)
    beta = rng.uniform(-30.0, 30.0)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if rng.random() < 0.35:
        gamma = rng.uniform(0.70, 1.45)
        lut = np.array([((idx / 255.0) ** gamma) * 255.0 for idx in range(256)], dtype=np.uint8)
        image = cv2.LUT(image, lut)

    if rng.random() < 0.20:
        k = rng.choice((3, 5))
        image = cv2.GaussianBlur(image, (k, k), 0)

    if rng.random() < 0.30:
        sigma = rng.uniform(2.0, 10.0)
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0.0, sigma, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.15:
        quality = rng.randint(45, 90)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                image = decoded

    return image


def preprocess(image: np.ndarray, img_size: int, augment: bool, rng: random.Random) -> torch.Tensor:
    if augment:
        image = apply_train_aug(image, rng)
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
    return tensor


class EyeStateClassificationDataset(Dataset):
    def __init__(self, samples: list[Sample], img_size: int, augment: bool, seed: int) -> None:
        self.samples = samples
        self.img_size = img_size
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]
        image = imread_color(sample.path)
        rng = random.Random(self.seed + idx * 7919)
        tensor = preprocess(image, self.img_size, self.augment, rng)
        return tensor, torch.tensor(sample.class_id, dtype=torch.long), str(sample.path)


class TinyEyeClassifier(nn.Module):
    def __init__(self, img_size: int = 160, width: int = 12) -> None:
        super().__init__()
        if img_size % 16 != 0:
            raise ValueError("--imgsz must be divisible by 16")
        c1 = width
        c2 = width * 2
        c3 = width * 3
        c4 = width * 4
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(c3, c4, 3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(c4, c4, 3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(img_size // 16),
        )
        self.classifier = nn.Linear(c4, len(CLASS_NAMES), bias=True)
        self.img_size = img_size
        self.width = width

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def make_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def make_loaders(splits: dict[str, list[Sample]], args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = EyeStateClassificationDataset(splits["train"], args.imgsz, augment=True, seed=args.seed)
    val_ds = EyeStateClassificationDataset(splits["val"], args.imgsz, augment=False, seed=args.seed)
    test_ds = EyeStateClassificationDataset(splits["test"], args.imgsz, augment=False, seed=args.seed)

    counts = class_counts(splits["train"])
    sample_weights = []
    for sample in splits["train"]:
        count = max(1, counts[sample.class_id])
        sample_weights.append(1.0 / count)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    return train_loader, val_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    total_loss = 0.0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels, _paths in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            preds = logits.argmax(dim=1)
            total_loss += float(loss.detach().cpu())
            total += int(labels.numel())
            for truth, pred in zip(labels.detach().cpu().tolist(), preds.detach().cpu().tolist()):
                confusion[truth, pred] += 1

    per_class = []
    for idx in range(len(CLASS_NAMES)):
        denom = int(confusion[idx].sum())
        per_class.append(float(confusion[idx, idx] / denom) if denom else math.nan)
    correct = int(np.trace(confusion))
    accuracy = float(correct / total) if total else 0.0
    valid_per_class = [value for value in per_class if not math.isnan(value)]
    balanced = float(sum(valid_per_class) / len(valid_per_class)) if valid_per_class else 0.0
    return {
        "loss": total_loss / total if total else 0.0,
        "total": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "per_class_accuracy": per_class,
        "confusion": confusion.tolist(),
    }


def save_manifest(path: Path, splits: dict[str, list[Sample]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class_id", "class_name", "path"])
        for split, samples in splits.items():
            for sample in samples:
                writer.writerow([split, sample.class_id, CLASS_NAMES[sample.class_id], sample.path.as_posix()])


def serializable_args(args: argparse.Namespace) -> dict:
    values = {}
    for key, value in vars(args).items():
        if callable(value):
            continue
        values[key] = value
    return values


def save_checkpoint(path: Path, model: TinyEyeClassifier, args: argparse.Namespace, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "img_size": model.img_size,
            "width": model.width,
            "classes": CLASS_NAMES,
            "args": serializable_args(args),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> TinyEyeClassifier:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TinyEyeClassifier(img_size=int(ckpt["img_size"]), width=int(ckpt["width"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def export_onnx(weight: Path, onnx_path: Path, opset: int, device: torch.device) -> None:
    model = load_checkpoint(weight, device)
    dummy = torch.zeros(1, 3, model.img_size, model.img_size, device=device)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[info] onnx={onnx_path}", flush=True)


def print_report(name: str, report: dict) -> None:
    print(f"[{name}] total={report['total']} acc={report['accuracy']:.3f} balanced={report['balanced_accuracy']:.3f}")
    for idx, class_name in enumerate(CLASS_NAMES):
        value = report["per_class_accuracy"][idx]
        text = "nan" if math.isnan(value) else f"{value:.3f}"
        row_total = sum(report["confusion"][idx])
        print(f"  {class_name}: acc={text} count={row_total} confusion={report['confusion'][idx]}")


def build_samples(args: argparse.Namespace) -> list[Sample]:
    open_samples = collect_class_samples([Path(args.open_dir)], 0, args.max_open, args.seed)
    closed_samples = collect_class_samples([Path(args.closed_dir)], 1, args.max_closed, args.seed)
    background_dirs = [Path(item) for item in args.background_dir]
    background_samples = collect_class_samples(background_dirs, 2, args.max_background, args.seed)
    samples = open_samples + closed_samples + background_samples
    counts = class_counts(samples)
    print(
        "[data] open={} closed={} background={} total={}".format(
            counts[0],
            counts[1],
            counts[2],
            sum(counts),
        ),
        flush=True,
    )
    if counts[0] == 0 or counts[1] == 0 or counts[2] == 0:
        raise SystemExit("classification training requires non-empty open, closed, and background classes")
    return samples


def train_command(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = build_samples(args)
    splits = split_samples(samples, args.val_ratio, args.test_ratio, args.seed)
    save_manifest(out_dir / "manifest.csv", splits)

    split_counts = {split: class_counts(items) for split, items in splits.items()}
    print(f"[data] split_counts={split_counts}", flush=True)

    device = make_device(args.device)
    model = TinyEyeClassifier(img_size=args.imgsz, width=args.width).to(device)
    train_loader, val_loader, test_loader = make_loaders(splits, args)

    train_counts = class_counts(splits["train"])
    weights = torch.tensor(
        [sum(train_counts) / max(1, len(CLASS_NAMES) * count) for count in train_counts],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_path = out_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_accuracy", "val_balanced_accuracy", "lr"])

    best_metric = -1.0
    best_report: dict | None = None
    last_report: dict | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels, _paths in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * int(labels.numel())
            seen += int(labels.numel())

        val_report = evaluate(model, val_loader, device) if len(splits["val"]) else evaluate(model, train_loader, device)
        last_report = val_report
        metric = float(val_report["balanced_accuracy"])
        train_loss = running_loss / max(1, seen)

        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, val_report["accuracy"], val_report["balanced_accuracy"], args.lr])

        print(
            f"epoch={epoch}/{args.epochs} loss={train_loss:.4f} "
            f"val_acc={val_report['accuracy']:.3f} val_balanced={val_report['balanced_accuracy']:.3f}",
            flush=True,
        )

        if metric >= best_metric:
            best_metric = metric
            best_report = val_report
            save_checkpoint(out_dir / "weights" / "best.pt", model, args, val_report)

        save_checkpoint(out_dir / "weights" / "last.pt", model, args, val_report)

    assert best_report is not None and last_report is not None
    best_model = load_checkpoint(out_dir / "weights" / "best.pt", device)
    train_report = evaluate(best_model, train_loader, device)
    val_report = evaluate(best_model, val_loader, device) if len(splits["val"]) else best_report
    test_report = evaluate(best_model, test_loader, device) if len(splits["test"]) else {"total": 0, "accuracy": 0.0, "balanced_accuracy": 0.0, "per_class_accuracy": [math.nan] * len(CLASS_NAMES), "confusion": [[0] * len(CLASS_NAMES) for _ in CLASS_NAMES]}
    print_report("train_best", train_report)
    print_report("val_best", val_report)
    print_report("test_best", test_report)

    metadata = {
        "model": "TinyEyeClassifier",
        "contract": {"input": [1, 3, args.imgsz, args.imgsz], "logits": [1, len(CLASS_NAMES)]},
        "classes": {idx: name for idx, name in enumerate(CLASS_NAMES)},
        "data": {
            "open_dir": args.open_dir,
            "closed_dir": args.closed_dir,
            "background_dir": args.background_dir,
            "split_counts": split_counts,
        },
        "best_metric": best_metric,
        "reports": {"train": train_report, "val": val_report, "test": test_report},
        "args": serializable_args(args),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.export_onnx:
        export_onnx(out_dir / "weights" / "best.pt", out_dir / "weights" / "best.onnx", args.opset, device)
    return 0


def inspect_command(args: argparse.Namespace) -> int:
    samples = build_samples(args)
    splits = split_samples(samples, args.val_ratio, args.test_ratio, args.seed)
    for split, items in splits.items():
        counts = class_counts(items)
        print(
            "[{}] open={} closed={} background={} total={}".format(
                split,
                counts[0],
                counts[1],
                counts[2],
                sum(counts),
            )
        )
    if args.manifest:
        save_manifest(Path(args.manifest), splits)
        print(f"[info] manifest={args.manifest}")
    return 0


def predict_paths(model: TinyEyeClassifier, paths: list[Path], img_size: int, device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        for path in paths:
            image = imread_color(path)
            tensor = preprocess(image, img_size, augment=False, rng=random.Random(0)).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            order = np.argsort(-probs)
            top = int(order[0])
            details = " ".join(f"{CLASS_NAMES[int(idx)]}={probs[int(idx)]:.3f}" for idx in order)
            print(f"{path} -> {CLASS_NAMES[top]} {details}")


def predict_command(args: argparse.Namespace) -> int:
    device = make_device(args.device)
    model = load_checkpoint(Path(args.weight), device)
    paths: list[Path] = []
    for raw in args.path:
        path = Path(raw)
        if path.is_dir():
            paths.extend(list_images(path))
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            paths.append(path)
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("no images to predict")
    predict_paths(model, paths, model.img_size, device)
    return 0


def export_command(args: argparse.Namespace) -> int:
    export_onnx(Path(args.weight), Path(args.onnx), args.opset, make_device(args.device))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    train.add_argument("--open-dir", default="data/eye_state_capture/eye_open")
    train.add_argument("--closed-dir", default="data/eye_state_capture/eye_closed")
    train.add_argument("--background-dir", action="append", default=["data/public_no_eye_wflw_v2"])
    train.add_argument("--max-open", type=int, default=0)
    train.add_argument("--max-closed", type=int, default=0)
    train.add_argument("--max-background", type=int, default=300)
    train.add_argument("--out-dir", default="runs/eye_state_classifier")
    train.add_argument("--name", default="eye_state_cls_v1")
    train.add_argument("--imgsz", type=int, default=160)
    train.add_argument("--width", type=int, default=12)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--batch", type=int, default=64)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--device", default="auto")
    train.add_argument("--lr", type=float, default=0.001)
    train.add_argument("--weight-decay", type=float, default=0.0001)
    train.add_argument("--grad-clip", type=float, default=5.0)
    train.add_argument("--val-ratio", type=float, default=0.15)
    train.add_argument("--test-ratio", type=float, default=0.15)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--export-onnx", action="store_true")
    train.add_argument("--opset", type=int, default=13)
    train.set_defaults(func=train_command)

    inspect = sub.add_parser("inspect")
    inspect.add_argument("--open-dir", default="data/eye_state_capture/eye_open")
    inspect.add_argument("--closed-dir", default="data/eye_state_capture/eye_closed")
    inspect.add_argument("--background-dir", action="append", default=["data/public_no_eye_wflw_v2"])
    inspect.add_argument("--max-open", type=int, default=0)
    inspect.add_argument("--max-closed", type=int, default=0)
    inspect.add_argument("--max-background", type=int, default=300)
    inspect.add_argument("--val-ratio", type=float, default=0.15)
    inspect.add_argument("--test-ratio", type=float, default=0.15)
    inspect.add_argument("--seed", type=int, default=42)
    inspect.add_argument("--manifest", default="")
    inspect.set_defaults(func=inspect_command)

    predict = sub.add_parser("predict")
    predict.add_argument("--weight", required=True)
    predict.add_argument("--path", action="append", required=True)
    predict.add_argument("--device", default="auto")
    predict.set_defaults(func=predict_command)

    export = sub.add_parser("export")
    export.add_argument("--weight", required=True)
    export.add_argument("--onnx", required=True)
    export.add_argument("--device", default="auto")
    export.add_argument("--opset", type=int, default=13)
    export.set_defaults(func=export_command)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
