#!/usr/bin/env python3
"""Train, export, and validate an ESP-DL-friendly direct eye-state detector.

Model contract:

    input:  images [1,3,320,320]
    output: boxes  [1,4,400]  # cx, cy, w, h in input pixels
            scores [1,2,400]  # eye_open, eye_closed confidences

The network intentionally uses only simple Conv/BatchNorm/ReLU blocks plus
static tensor arithmetic in the decode path. It does not use YOLO heads, DFL,
graph NMS, face detection, landmarks, cascades, or manual ROI logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
CLASS_NAMES = {0: "eye_open", 1: "eye_closed"}


@dataclass(frozen=True)
class GroundTruth:
    class_id: int
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DirectEyeDetector(nn.Module):
    """Small fixed-grid detector with static 20x20 candidate outputs."""

    def __init__(self, img_size: int = 320, grid_size: int = 20, width: int = 32, depth: str = "standard") -> None:
        super().__init__()
        if img_size % grid_size != 0:
            raise ValueError("img_size must be divisible by grid_size")
        if depth not in {"standard", "lite", "lite_s8"}:
            raise ValueError("depth must be 'standard', 'lite', or 'lite_s8'")
        if depth in {"standard", "lite"} and img_size // grid_size != 16:
            raise ValueError("standard/lite depth expects stride-16 output, e.g. imgsz=320 grid=20")
        if depth == "lite_s8" and img_size // grid_size != 8:
            raise ValueError("lite_s8 depth expects stride-8 output, e.g. imgsz=320 grid=40")

        c1 = width
        c2 = width * 2
        c3 = width * 3
        c4 = width * 5
        self.img_size = int(img_size)
        self.grid_size = int(grid_size)
        self.depth = depth
        self.stride = float(img_size // grid_size)
        self.max_box = float(img_size)

        if depth == "lite_s8":
            c4 = width * 4
            self.backbone = nn.Sequential(
                ConvBnRelu(3, c1, stride=2),
                ConvBnRelu(c1, c2, stride=2),
                ConvBnRelu(c2, c3, stride=2),
                ConvBnRelu(c3, c4),
                ConvBnRelu(c4, c4),
            )
            self.head = nn.Sequential(
                nn.Conv2d(c4, 7, 1),
            )
        elif depth == "lite":
            c4 = width * 4
            self.backbone = nn.Sequential(
                ConvBnRelu(3, c1, stride=2),
                ConvBnRelu(c1, c2, stride=2),
                ConvBnRelu(c2, c3, stride=2),
                ConvBnRelu(c3, c4, stride=2),
                ConvBnRelu(c4, c4),
            )
            self.head = nn.Sequential(
                nn.Conv2d(c4, 7, 1),
            )
        else:
            self.backbone = nn.Sequential(
                ConvBnRelu(3, c1, stride=2),
                ConvBnRelu(c1, c1),
                ConvBnRelu(c1, c2, stride=2),
                ConvBnRelu(c2, c2),
                ConvBnRelu(c2, c3, stride=2),
                ConvBnRelu(c3, c3),
                ConvBnRelu(c3, c4, stride=2),
                ConvBnRelu(c4, c4),
                ConvBnRelu(c4, c4),
                ConvBnRelu(c4, c4),
            )
            self.head = nn.Sequential(
                ConvBnRelu(c4, c3),
                nn.Conv2d(c3, 7, 1),
            )

        gy, gx = torch.meshgrid(
            torch.arange(grid_size, dtype=torch.float32),
            torch.arange(grid_size, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("grid_x", gx.view(1, 1, grid_size, grid_size), persistent=False)
        self.register_buffer("grid_y", gy.view(1, 1, grid_size, grid_size), persistent=False)

    def forward_raw(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))

    def decode(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        box_raw = raw[:, 0:4]
        obj_logit = raw[:, 4:5]
        cls_logit = raw[:, 5:7]

        center = torch.sigmoid(box_raw[:, 0:2])
        size = torch.sigmoid(box_raw[:, 2:4])

        cx = (self.grid_x + center[:, 0:1]) * self.stride
        cy = (self.grid_y + center[:, 1:2]) * self.stride
        bw = size[:, 0:1] * self.max_box
        bh = size[:, 1:2] * self.max_box
        boxes = torch.cat((cx, cy, bw, bh), dim=1).flatten(2)

        scores = torch.sigmoid(obj_logit) * torch.sigmoid(cls_logit)
        scores = scores.flatten(2)
        return boxes, scores

    def decode_static_batch1(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        box_raw = raw[:, 0:4]
        obj_logit = raw[:, 4:5]
        cls_logit = raw[:, 5:7]

        center = torch.sigmoid(box_raw[:, 0:2])
        size = torch.sigmoid(box_raw[:, 2:4])

        cx = (self.grid_x + center[:, 0:1]) * self.stride
        cy = (self.grid_y + center[:, 1:2]) * self.stride
        bw = size[:, 0:1] * self.max_box
        bh = size[:, 1:2] * self.max_box
        boxes = torch.cat((cx, cy, bw, bh), dim=1).reshape(1, 4, self.grid_size * self.grid_size)

        scores = torch.sigmoid(obj_logit) * torch.sigmoid(cls_logit)
        scores = scores.reshape(1, 2, self.grid_size * self.grid_size)
        return boxes, scores

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decode(self.forward_raw(images))


class StaticBatch1Export(nn.Module):
    def __init__(self, model: DirectEyeDetector) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.model.forward_raw(images)
        return self.model.decode_static_batch1(raw)


class RawHeadExport(nn.Module):
    def __init__(self, model: DirectEyeDetector) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.forward_raw(images)


class EyeYoloDataset(Dataset):
    def __init__(
        self,
        data_yaml: Path,
        split: str,
        img_size: int,
        grid_size: int,
        augment: bool = False,
        hard_negative_prob: float = 0.0,
        positive_zoom_prob: float = 0.0,
    ) -> None:
        self.data_yaml = data_yaml
        self.split = split
        self.img_size = int(img_size)
        self.grid_size = int(grid_size)
        self.augment = augment
        self.hard_negative_prob = float(hard_negative_prob)
        self.positive_zoom_prob = float(positive_zoom_prob)

        cfg = load_yaml(data_yaml)
        root = resolve_dataset_dir(data_yaml, cfg.get("path", data_yaml.parent))
        image_value = cfg.get(split)
        if not image_value:
            raise ValueError(f"{data_yaml} does not define split '{split}'")
        self.image_root = resolve_dataset_dir(data_yaml, Path(root) / image_value)
        self.label_root = Path(str(self.image_root).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
        if not self.image_root.exists():
            raise FileNotFoundError(self.image_root)

        self.images = list_images([self.image_root])
        if not self.images:
            raise ValueError(f"no images found in {self.image_root}")
        self.label_paths = [image_label_path(path, self.image_root, self.label_root) for path in self.images]
        self.has_closed = [label_has_class(path, 1) for path in self.label_paths]
        self.has_labels = [bool(read_yolo_labels(path)) for path in self.label_paths]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        label_path = self.label_paths[index]
        image = imread_color(image_path)
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = read_yolo_labels(label_path)

        if self.augment and labels and self.hard_negative_prob > 0.0 and random.random() < self.hard_negative_prob:
            image, labels = make_hard_negative_crop(image, labels)
        elif self.augment and labels and self.positive_zoom_prob > 0.0 and random.random() < self.positive_zoom_prob:
            image, labels = make_positive_zoom_crop(image, labels)

        if self.augment:
            image, labels = augment_image(image, labels)

        height, width = image.shape[:2]
        if width != self.img_size or height != self.img_size:
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        targets = build_targets(labels, self.img_size, self.grid_size)
        return tensor, targets["obj"], targets["cls"], targets["box"], str(image_path)


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


def list_images(paths: Iterable[Path]) -> list[Path]:
    images: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
        elif path.is_dir():
            images.extend(
                p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
    return sorted(images)


def image_label_path(image_path: Path, image_root: Path, label_root: Path) -> Path:
    rel = image_path.relative_to(image_root)
    return (label_root / rel).with_suffix(".txt")


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    labels: list[tuple[int, float, float, float, float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = parts[:5]
        class_id = int(float(cls))
        if class_id not in CLASS_NAMES:
            continue
        labels.append((class_id, float(cx), float(cy), float(bw), float(bh)))
    return labels


def imread_color(path: Path | str) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        data = np.empty((0,), dtype=np.uint8)
    if data.size:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def label_has_class(path: Path, class_id: int) -> bool:
    return any(label[0] == class_id for label in read_yolo_labels(path))


def augment_image(
    image: np.ndarray,
    labels: list[tuple[int, float, float, float, float]],
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    out = image
    updated = list(labels)

    if random.random() < 0.5:
        out = np.ascontiguousarray(out[:, ::-1, :])
        updated = [(cls, 1.0 - cx, cy, bw, bh) for cls, cx, cy, bw, bh in updated]

    if random.random() < 0.8:
        gain = random.uniform(0.75, 1.25)
        bias = random.uniform(-18.0, 18.0)
        out = np.clip(out.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

    if random.random() < 0.25:
        noise = np.random.normal(0.0, 4.0, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out, updated


def labels_to_xyxy(
    labels: list[tuple[int, float, float, float, float]],
    width: int,
    height: int,
) -> list[tuple[int, tuple[float, float, float, float]]]:
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for class_id, cx, cy, bw, bh in labels:
        px = float(np.clip(cx, 0.0, 1.0)) * width
        py = float(np.clip(cy, 0.0, 1.0)) * height
        pw = float(np.clip(bw, 0.0, 1.0)) * width
        ph = float(np.clip(bh, 0.0, 1.0)) * height
        boxes.append((int(class_id), (px - pw / 2.0, py - ph / 2.0, px + pw / 2.0, py + ph / 2.0)))
    return boxes


def crop_and_relabel(
    image: np.ndarray,
    labels: list[tuple[int, float, float, float, float]],
    crop_box: tuple[int, int, int, int],
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    x1, y1, x2, y2 = crop_box
    crop = image[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]
    if crop_w <= 0 or crop_h <= 0:
        return image, labels

    relabeled: list[tuple[int, float, float, float, float]] = []
    for class_id, box in labels_to_xyxy(labels, image.shape[1], image.shape[0]):
        ix1 = max(box[0], float(x1))
        iy1 = max(box[1], float(y1))
        ix2 = min(box[2], float(x2))
        iy2 = min(box[3], float(y2))
        if ix2 - ix1 < 1.0 or iy2 - iy1 < 1.0:
            continue
        cx = ((ix1 + ix2) * 0.5 - x1) / crop_w
        cy = ((iy1 + iy2) * 0.5 - y1) / crop_h
        bw = (ix2 - ix1) / crop_w
        bh = (iy2 - iy1) / crop_h
        relabeled.append((class_id, cx, cy, bw, bh))
    return crop, relabeled


def make_hard_negative_crop(
    image: np.ndarray,
    labels: list[tuple[int, float, float, float, float]],
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    if not labels:
        return image, labels

    height, width = image.shape[:2]
    boxes = []
    for _class_id, cx, cy, bw, bh in labels:
        px = float(np.clip(cx, 0.0, 1.0)) * width
        py = float(np.clip(cy, 0.0, 1.0)) * height
        pw = float(np.clip(bw, 0.0, 1.0)) * width
        ph = float(np.clip(bh, 0.0, 1.0)) * height
        boxes.append((px - pw / 2.0, py - ph / 2.0, px + pw / 2.0, py + ph / 2.0))

    eye_left = min(box[0] for box in boxes)
    eye_top = min(box[1] for box in boxes)
    eye_right = max(box[2] for box in boxes)
    eye_bottom = max(box[3] for box in boxes)
    eye_center_x = 0.5 * (eye_left + eye_right)

    min_crop_w = max(48, int(width * 0.45))
    max_crop_w = max(min_crop_w, int(width * 0.95))
    min_crop_h = max(48, int(height * 0.30))
    max_crop_h = max(min_crop_h, int(height * 0.75))

    for _attempt in range(12):
        crop_w = random.randint(min_crop_w, max_crop_w)
        crop_h = random.randint(min_crop_h, max_crop_h)

        max_x1 = width - crop_w
        max_y1 = height - crop_h
        min_y1 = int(eye_bottom + 1)
        if max_x1 < 0 or max_y1 < min_y1:
            continue

        x_low = max(0, int(eye_center_x - crop_w * 0.75))
        x_high = min(max_x1, int(eye_center_x - crop_w * 0.25))
        y_low = max(0, min_y1)
        y_high = min(max_y1, int(eye_bottom + height * 0.45))

        if x_high < x_low or y_high < y_low:
            continue

        x1 = random.randint(x_low, x_high)
        y1 = random.randint(y_low, y_high)
        crop_box = (float(x1), float(y1), float(x1 + crop_w), float(y1 + crop_h))
        if any(box_iou(crop_box, box) > 0.0 for box in boxes):
            continue

        crop = image[y1 : y1 + crop_h, x1 : x1 + crop_w]
        if crop.size == 0:
            continue
        return crop, []

    return image, labels


def make_positive_zoom_crop(
    image: np.ndarray,
    labels: list[tuple[int, float, float, float, float]],
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    if not labels:
        return image, labels

    height, width = image.shape[:2]
    boxes = labels_to_xyxy(labels, width, height)
    if not boxes:
        return image, labels

    # Prefer closed eyes slightly so the model sees close-up blink examples more often.
    closed_boxes = [item for item in boxes if item[0] == 1]
    target_class, target_box = random.choice(closed_boxes or boxes)
    tx1, ty1, tx2, ty2 = target_box
    target_cx = 0.5 * (tx1 + tx2)
    target_cy = 0.5 * (ty1 + ty2)
    target_w = max(1.0, tx2 - tx1)
    target_h = max(1.0, ty2 - ty1)

    for _attempt in range(10):
        scale = random.uniform(2.2, 4.8)
        crop_w = min(width, max(64, int(target_w * scale)))
        crop_h = min(height, max(64, int(target_h * scale)))
        if crop_w >= width and crop_h >= height:
            return image, labels

        jitter_x = random.uniform(-0.25, 0.25) * target_w
        jitter_y = random.uniform(-0.25, 0.25) * target_h
        x1 = int(round(target_cx + jitter_x - crop_w * 0.5))
        y1 = int(round(target_cy + jitter_y - crop_h * 0.5))
        x1 = max(0, min(width - crop_w, x1))
        y1 = max(0, min(height - crop_h, y1))
        crop, relabeled = crop_and_relabel(image, labels, (x1, y1, x1 + crop_w, y1 + crop_h))
        if not relabeled:
            continue
        if not any(item[0] == target_class for item in relabeled):
            continue
        return crop, relabeled

    return image, labels


def build_targets(
    labels: list[tuple[int, float, float, float, float]],
    img_size: int,
    grid_size: int,
) -> dict[str, torch.Tensor]:
    obj = torch.zeros((1, grid_size, grid_size), dtype=torch.float32)
    cls_target = torch.zeros((grid_size, grid_size), dtype=torch.long)
    box = torch.zeros((4, grid_size, grid_size), dtype=torch.float32)
    area = torch.zeros((grid_size, grid_size), dtype=torch.float32)

    for class_id, cx, cy, bw, bh in labels:
        px = float(np.clip(cx, 0.0, 1.0)) * img_size
        py = float(np.clip(cy, 0.0, 1.0)) * img_size
        pw = float(np.clip(bw, 0.0, 1.0)) * img_size
        ph = float(np.clip(bh, 0.0, 1.0)) * img_size
        gx = min(grid_size - 1, max(0, int(px / img_size * grid_size)))
        gy = min(grid_size - 1, max(0, int(py / img_size * grid_size)))
        current_area = pw * ph
        if obj[0, gy, gx] > 0 and current_area <= area[gy, gx]:
            continue
        obj[0, gy, gx] = 1.0
        cls_target[gy, gx] = int(class_id)
        box[:, gy, gx] = torch.tensor((px, py, pw, ph), dtype=torch.float32)
        area[gy, gx] = current_area

    return {"obj": obj, "cls": cls_target, "box": box}


def xywh_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    cx, cy, w, h = box
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def make_model(args: argparse.Namespace | dict) -> DirectEyeDetector:
    values = vars(args) if isinstance(args, argparse.Namespace) else args
    return DirectEyeDetector(
        img_size=int(values.get("imgsz", values.get("img_size", 320))),
        grid_size=int(values.get("grid", values.get("grid_size", 20))),
        width=int(values.get("width", 32)),
        depth=str(values.get("depth", "standard")),
    )


def device_from_arg(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def make_loader(dataset: EyeYoloDataset, args: argparse.Namespace, train: bool) -> DataLoader:
    sampler = None
    shuffle = train
    if train and (args.closed_sample_weight > 1.0 or args.negative_sample_weight > 1.0):
        weights = []
        for closed, has_labels in zip(dataset.has_closed, dataset.has_labels):
            weight = 1.0
            if closed:
                weight *= args.closed_sample_weight
            if not has_labels:
                weight *= args.negative_sample_weight
            weights.append(weight)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False,
    )


def compute_loss(
    model: DirectEyeDetector,
    images: torch.Tensor,
    target_obj: torch.Tensor,
    target_cls: torch.Tensor,
    target_box: torch.Tensor,
    class_weight: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw = model.forward_raw(images)
    boxes, _ = model.decode(raw)
    obj_logits = raw[:, 4:5]
    cls_logits = raw[:, 5:7]

    obj_loss_map = F.binary_cross_entropy_with_logits(obj_logits, target_obj, reduction="none")
    obj_prob = torch.sigmoid(obj_logits)
    p_t = target_obj * obj_prob + (1.0 - target_obj) * (1.0 - obj_prob)
    focal = torch.pow(1.0 - p_t, args.obj_focal_gamma) if args.obj_focal_gamma > 0.0 else 1.0
    obj_weights = torch.where(
        target_obj > 0.5,
        torch.full_like(target_obj, args.obj_pos_weight),
        torch.full_like(target_obj, args.obj_neg_weight),
    )
    weighted_obj = obj_loss_map * focal * obj_weights
    loss_obj = weighted_obj.sum() / obj_weights.sum().clamp_min(1.0)
    if args.obj_hard_neg_topk > 0 and args.obj_hard_neg_weight > 0.0:
        hard_neg = (obj_loss_map * focal).masked_fill(target_obj > 0.5, -1.0).flatten(1)
        topk = min(int(args.obj_hard_neg_topk), hard_neg.shape[1])
        hard_values = torch.topk(hard_neg, k=topk, dim=1).values
        hard_values = hard_values[hard_values > 0.0]
        if hard_values.numel() > 0:
            loss_obj = loss_obj + args.obj_hard_neg_weight * hard_values.mean()

    pos = target_obj[:, 0].bool()
    if pos.any():
        cls_pos = cls_logits.permute(0, 2, 3, 1)[pos]
        cls_target_pos = target_cls[pos]
        loss_cls = F.cross_entropy(cls_pos, cls_target_pos, weight=class_weight)

        decoded_grid = boxes.view(images.shape[0], 4, model.grid_size, model.grid_size)
        pred_box_pos = decoded_grid.permute(0, 2, 3, 1)[pos] / float(model.img_size)
        target_box_pos = target_box.permute(0, 2, 3, 1)[pos] / float(model.img_size)
        loss_box = F.smooth_l1_loss(pred_box_pos, target_box_pos, beta=0.05)
    else:
        loss_cls = raw.sum() * 0.0
        loss_box = raw.sum() * 0.0

    total = args.obj_gain * loss_obj + args.cls_gain * loss_cls + args.box_gain * loss_box
    return total, {
        "obj": float(loss_obj.detach().cpu()),
        "cls": float(loss_cls.detach().cpu()),
        "box": float(loss_box.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def train_command(args: argparse.Namespace) -> int:
    set_reproducible(args.seed)
    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)

    device = device_from_arg(args.device)
    train_set = EyeYoloDataset(
        Path(args.data),
        "train",
        args.imgsz,
        args.grid,
        augment=True,
        hard_negative_prob=args.hard_negative_prob,
        positive_zoom_prob=args.positive_zoom_prob,
    )
    val_set = EyeYoloDataset(Path(args.data), "val", args.imgsz, args.grid, augment=False)
    train_loader = make_loader(train_set, args, train=True)
    val_loader = make_loader(val_set, args, train=False)

    model = make_model(args).to(device)
    if args.init_weight:
        checkpoint = torch.load(Path(args.init_weight), map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"[info] init_weight={args.init_weight}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    class_weight = torch.tensor([1.0, args.closed_class_weight], dtype=torch.float32, device=device)

    serializable_args = {key: value for key, value in vars(args).items() if key != "func"}
    metadata = {
        "model": "DirectEyeDetector",
        "contract": {"input": [1, 3, args.imgsz, args.imgsz], "boxes": [1, 4, args.grid * args.grid], "scores": [1, 2, args.grid * args.grid]},
        "data": args.data,
        "classes": CLASS_NAMES,
        "args": serializable_args,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    best_metric = float("-inf")
    best_epoch = 0
    history_path = out_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_obj",
                "train_cls",
                "train_box",
                "val_positive_rate",
                "val_eye_open_rate",
                "val_eye_closed_rate",
                "val_no_eye_pass_rate",
                "val_precision",
                "val_false_positive_total",
                "val_false_positive_per_image",
                "metric",
                "lr",
            ]
        )

    print(f"[info] device={device}", flush=True)
    print(f"[info] train_images={len(train_set)} val_images={len(val_set)}", flush=True)
    print(f"[info] output={out_dir}", flush=True)

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        accum = {"total": 0.0, "obj": 0.0, "cls": 0.0, "box": 0.0}
        seen = 0
        for images, target_obj, target_cls, target_box, _paths in train_loader:
            images = images.to(device, non_blocking=True)
            target_obj = target_obj.to(device, non_blocking=True)
            target_cls = target_cls.to(device, non_blocking=True)
            target_box = target_box.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss, parts = compute_loss(model, images, target_obj, target_cls, target_box, class_weight, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_size = images.shape[0]
            seen += batch_size
            for key in accum:
                accum[key] += parts[key] * batch_size

        scheduler.step()
        train_parts = {key: value / max(1, seen) for key, value in accum.items()}
        val_report = evaluate_model(model, val_loader, device, args.conf, args.min_iou, args.max_det)
        metric = (
            0.50 * val_report["positive_rate"]
            + 0.20 * val_report["eye_closed_rate"]
            + 0.15 * val_report["no_eye_pass_rate"]
            + 0.15 * val_report["precision"]
            - 0.01 * val_report["false_positive_per_image"]
        )
        if val_report["positive_rate"] < args.min_metric_positive_rate:
            metric -= 1.0
        if val_report["eye_closed_rate"] < args.min_metric_closed_rate:
            metric -= 0.5
        if val_report["precision"] < args.min_metric_precision:
            metric -= 1.0
        is_best = metric > best_metric
        if is_best:
            best_metric = metric
            best_epoch = epoch
            save_checkpoint(out_dir / "weights" / "best.pt", model, args, epoch, best_metric)
        save_checkpoint(out_dir / "weights" / "last.pt", model, args, epoch, best_metric)

        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    train_parts["total"],
                    train_parts["obj"],
                    train_parts["cls"],
                    train_parts["box"],
                    val_report["positive_rate"],
                    val_report["eye_open_rate"],
                    val_report["eye_closed_rate"],
                    val_report["no_eye_pass_rate"],
                    val_report["precision"],
                    val_report["false_positive_total"],
                    val_report["false_positive_per_image"],
                    metric,
                    optimizer.param_groups[0]["lr"],
                ]
            )

        elapsed = time.time() - started
        print(
            "epoch={}/{} loss={:.4f} obj={:.4f} cls={:.4f} box={:.4f} "
            "val_pos={}/{}={:.1f}% val_open={}/{}={:.1f}% "
            "val_closed={}/{}={:.1f}% val_noeye={}/{}={:.1f}% "
            "precision={:.1f}% fp/img={:.2f} "
            "metric={:.4f} best_epoch={} time={:.1f}s{}".format(
                epoch,
                args.epochs,
                train_parts["total"],
                train_parts["obj"],
                train_parts["cls"],
                train_parts["box"],
                val_report["positive_pass"],
                val_report["positive_total"],
                100.0 * val_report["positive_rate"],
                val_report["eye_open_pass"],
                val_report["eye_open_total"],
                100.0 * val_report["eye_open_rate"],
                val_report["eye_closed_pass"],
                val_report["eye_closed_total"],
                100.0 * val_report["eye_closed_rate"],
                val_report["no_eye_pass"],
                val_report["no_eye_total"],
                100.0 * val_report["no_eye_pass_rate"],
                100.0 * val_report["precision"],
                val_report["false_positive_per_image"],
                metric,
                best_epoch,
                elapsed,
                " best" if is_best else "",
            ),
            flush=True,
        )

    best = out_dir / "weights" / "best.pt"
    if args.export_onnx:
        export_checkpoint(best, out_dir / "weights" / "best.onnx", args.opset)
    return 0


def save_checkpoint(
    path: Path,
    model: DirectEyeDetector,
    args: argparse.Namespace,
    epoch: int,
    metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_args": {"imgsz": args.imgsz, "grid": args.grid, "width": args.width, "depth": args.depth},
            "epoch": epoch,
            "metric": metric,
            "classes": CLASS_NAMES,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> DirectEyeDetector:
    checkpoint = torch.load(path, map_location=device)
    model_args = checkpoint.get("model_args", {})
    model = make_model(model_args).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def export_checkpoint(weight: Path, onnx_path: Path, opset: int, raw: bool = False) -> None:
    device = torch.device("cpu")
    checkpoint = torch.load(weight, map_location=device)
    model_args = checkpoint.get("model_args", {})
    model = make_model(model_args).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    img_size = int(model_args.get("imgsz", model_args.get("img_size", 320)))
    dummy = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_model = RawHeadExport(model).eval() if raw else StaticBatch1Export(model).eval()
    output_names = ["raw"] if raw else ["boxes", "scores"]
    torch.onnx.export(
        export_model,
        dummy,
        str(onnx_path),
        input_names=["images"],
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    print(f"[info] onnx={onnx_path} mode={'raw' if raw else 'decoded'}", flush=True)


def export_command(args: argparse.Namespace) -> int:
    export_checkpoint(Path(args.weight), Path(args.onnx), args.opset, raw=args.raw)
    return 0


def validate_command(args: argparse.Namespace) -> int:
    device = device_from_arg(args.device)
    model = load_checkpoint(Path(args.weight), device)
    dataset = EyeYoloDataset(Path(args.data), args.split, model.img_size, model.grid_size, augment=False)
    loader_args = argparse.Namespace(batch=args.batch, workers=args.workers, closed_sample_weight=1.0)
    loader = make_loader(dataset, loader_args, train=False)
    report = evaluate_model(model, loader, device, args.conf, args.min_iou, args.max_det)

    print_report(f"{args.split}", report)

    generated_dir = Path(args.out_dir) / "generated_no_eye"
    generated_images = make_negative_images(generated_dir, model.img_size)
    generated_report = evaluate_unlabeled_images(model, generated_images, device, args.conf, args.max_det)
    print_negative_report("generated_no_eye", generated_report)

    if args.qualitative_dir:
        paths = list_images([Path(args.qualitative_dir)])
        qualitative_report = evaluate_unlabeled_images(model, paths, device, args.conf, args.max_det)
        print_negative_report("external_unlabeled_no_detection", qualitative_report)
        print("[external_unlabeled_top]")
        for item in qualitative_report["items"]:
            if item["predictions"]:
                top = item["predictions"][0]
                print(
                    f"  {item['path']} -> {CLASS_NAMES[top.class_id]} "
                    f"conf={top.confidence:.3f} xyxy={tuple(round(v, 1) for v in top.xyxy)}"
                )
            else:
                print(f"  {item['path']} -> no_detection")

    return 0


def evaluate_model(
    model: DirectEyeDetector,
    loader: DataLoader,
    device: torch.device,
    conf: float,
    min_iou: float,
    max_det: int,
) -> dict[str, float | int]:
    totals = {
        "image_total": 0,
        "positive_total": 0,
        "positive_pass": 0,
        "eye_open_total": 0,
        "eye_open_pass": 0,
        "eye_closed_total": 0,
        "eye_closed_pass": 0,
        "no_eye_total": 0,
        "no_eye_pass": 0,
        "prediction_total": 0,
        "false_positive_total": 0,
    }
    model.eval()
    with torch.no_grad():
        for images, target_obj, target_cls, target_box, paths in loader:
            images = images.to(device)
            boxes, scores = model(images)
            batch_predictions = predictions_from_tensors(boxes.cpu(), scores.cpu(), conf, max_det)
            truths = truths_from_image_paths(paths, model.img_size)
            for image_truths, predictions in zip(truths, batch_predictions):
                totals["image_total"] += 1
                totals["prediction_total"] += len(predictions)
                if not image_truths:
                    totals["no_eye_total"] += 1
                    if not predictions:
                        totals["no_eye_pass"] += 1
                    totals["false_positive_total"] += len(predictions)
                    continue

                matched_predictions: set[int] = set()
                for truth in image_truths:
                    totals["positive_total"] += 1
                    if truth.class_id == 0:
                        totals["eye_open_total"] += 1
                    else:
                        totals["eye_closed_total"] += 1
                    match = find_match(truth, predictions, matched_predictions, min_iou)
                    if match is None:
                        continue
                    matched_predictions.add(match)
                    totals["positive_pass"] += 1
                    if truth.class_id == 0:
                        totals["eye_open_pass"] += 1
                    else:
                        totals["eye_closed_pass"] += 1
                totals["false_positive_total"] += max(0, len(predictions) - len(matched_predictions))

    totals["positive_rate"] = safe_rate(totals["positive_pass"], totals["positive_total"])
    totals["eye_open_rate"] = safe_rate(totals["eye_open_pass"], totals["eye_open_total"])
    totals["eye_closed_rate"] = safe_rate(totals["eye_closed_pass"], totals["eye_closed_total"])
    totals["no_eye_pass_rate"] = safe_rate(totals["no_eye_pass"], totals["no_eye_total"])
    totals["precision"] = safe_rate(totals["positive_pass"], totals["positive_pass"] + totals["false_positive_total"])
    totals["false_positive_per_image"] = safe_rate(totals["false_positive_total"], totals["image_total"])
    return totals


def truths_from_image_paths(paths: list[str], img_size: int) -> list[list[GroundTruth]]:
    batch: list[list[GroundTruth]] = []
    for raw_path in paths:
        image_path = Path(raw_path)
        label_path = label_path_from_image_path(image_path)
        truths: list[GroundTruth] = []
        for class_id, cx, cy, bw, bh in read_yolo_labels(label_path):
            center_x = cx * img_size
            center_y = cy * img_size
            box_w = bw * img_size
            box_h = bh * img_size
            truths.append(GroundTruth(class_id, xywh_to_xyxy((center_x, center_y, box_w, box_h))))
        batch.append(truths)
    return batch


def label_path_from_image_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def truths_from_targets(
    target_obj: torch.Tensor,
    target_cls: torch.Tensor,
    target_box: torch.Tensor,
) -> list[list[GroundTruth]]:
    batch: list[list[GroundTruth]] = []
    for idx in range(target_obj.shape[0]):
        truths: list[GroundTruth] = []
        pos = target_obj[idx, 0] > 0.5
        ys, xs = torch.where(pos)
        for y, x in zip(ys.tolist(), xs.tolist()):
            cx, cy, bw, bh = target_box[idx, :, y, x].tolist()
            truths.append(GroundTruth(int(target_cls[idx, y, x].item()), xywh_to_xyxy((cx, cy, bw, bh))))
        batch.append(truths)
    return batch


def predictions_from_tensors(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    conf: float,
    max_det: int,
) -> list[list[Prediction]]:
    batch: list[list[Prediction]] = []
    for idx in range(boxes.shape[0]):
        candidates: list[Prediction] = []
        image_boxes = boxes[idx].transpose(0, 1).numpy()
        image_scores = scores[idx].transpose(0, 1).numpy()
        for candidate_idx in range(image_boxes.shape[0]):
            class_id = int(np.argmax(image_scores[candidate_idx]))
            confidence = float(image_scores[candidate_idx, class_id])
            if confidence < conf:
                continue
            xyxy = xywh_to_xyxy(tuple(float(v) for v in image_boxes[candidate_idx]))
            candidates.append(Prediction(class_id, confidence, xyxy))
        batch.append(nms(candidates, 0.45, max_det))
    return batch


def nms(predictions: list[Prediction], threshold: float, max_det: int) -> list[Prediction]:
    selected: list[Prediction] = []
    for pred in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        if len(selected) >= max_det:
            break
        if any(pred.class_id == kept.class_id and box_iou(pred.xyxy, kept.xyxy) > threshold for kept in selected):
            continue
        selected.append(pred)
    return selected


def find_match(
    truth: GroundTruth,
    predictions: list[Prediction],
    used: set[int],
    min_iou: float,
) -> int | None:
    best_idx = None
    best_iou = 0.0
    for idx, pred in enumerate(predictions):
        if idx in used or pred.class_id != truth.class_id:
            continue
        iou = box_iou(truth.xyxy, pred.xyxy)
        if iou >= min_iou and iou > best_iou:
            best_idx = idx
            best_iou = iou
    return best_idx


def make_negative_images(out_dir: Path, size: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    images: dict[str, np.ndarray] = {}
    images["black"] = np.zeros((size, size, 3), dtype=np.uint8)
    images["white"] = np.full((size, size, 3), 255, dtype=np.uint8)
    images["gray"] = np.full((size, size, 3), 128, dtype=np.uint8)
    grad = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    images["gradient_x"] = cv2.merge((grad, grad, grad))
    images["gradient_y"] = cv2.merge((grad.T, grad.T, grad.T))
    checker = ((np.indices((size, size)).sum(axis=0) // 16) % 2 * 255).astype(np.uint8)
    images["checker"] = cv2.merge((checker, checker, checker))
    images["noise"] = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    images["dark_noise"] = rng.integers(0, 48, (size, size, 3), dtype=np.uint8)
    images["low_contrast"] = rng.integers(96, 160, (size, size, 3), dtype=np.uint8)
    blocks = np.zeros((size, size, 3), dtype=np.uint8)
    half = size // 2
    blocks[:half, :half] = (40, 80, 180)
    blocks[:half, half:] = (180, 120, 40)
    blocks[half:, :half] = (60, 160, 80)
    blocks[half:, half:] = (180, 180, 180)
    images["color_blocks"] = blocks
    text = np.full((size, size, 3), 220, dtype=np.uint8)
    cv2.putText(text, "NO EYE", (size // 8, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    images["text"] = text

    paths: list[Path] = []
    for name, image in images.items():
        path = out_dir / f"{name}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        paths.append(path)
    return paths


def evaluate_unlabeled_images(
    model: DirectEyeDetector,
    paths: list[Path],
    device: torch.device,
    conf: float,
    max_det: int,
) -> dict:
    items = []
    passed = 0
    model.eval()
    with torch.no_grad():
        for path in paths:
            image = imread_color(path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (model.img_size, model.img_size), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
            boxes, scores = model(tensor.unsqueeze(0).to(device))
            preds = predictions_from_tensors(boxes.cpu(), scores.cpu(), conf, max_det)[0]
            if not preds:
                passed += 1
            items.append({"path": str(path), "predictions": preds})
    return {"pass": passed, "total": len(items), "items": items}


def print_report(name: str, report: dict) -> None:
    print(f"[{name}]")
    print(
        "positive: {}/{} = {:.1f}%".format(
            report["positive_pass"], report["positive_total"], 100.0 * report["positive_rate"]
        )
    )
    print(
        "eye_open: {}/{} = {:.1f}%".format(
            report["eye_open_pass"], report["eye_open_total"], 100.0 * report["eye_open_rate"]
        )
    )
    print(
        "eye_closed: {}/{} = {:.1f}%".format(
            report["eye_closed_pass"], report["eye_closed_total"], 100.0 * report["eye_closed_rate"]
        )
    )
    print(
        "no_eye: {}/{} = {:.1f}%".format(
            report["no_eye_pass"], report["no_eye_total"], 100.0 * report["no_eye_pass_rate"]
        )
    )


def print_negative_report(name: str, report: dict) -> None:
    rate = safe_rate(report["pass"], report["total"])
    print(f"[{name}] no_detection: {report['pass']}/{report['total']} = {100.0 * rate:.1f}%")


def safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="train the direct detector")
    train.add_argument("--data", required=True)
    train.add_argument("--out-dir", default="runs/eye_state_direct")
    train.add_argument("--name", default="direct_eye_state_v1")
    train.add_argument("--imgsz", type=int, default=320)
    train.add_argument("--grid", type=int, default=20)
    train.add_argument("--width", type=int, default=32)
    train.add_argument("--depth", choices=("standard", "lite", "lite_s8"), default="standard")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch", type=int, default=32)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--device", default="auto")
    train.add_argument("--lr", type=float, default=1.0e-3)
    train.add_argument("--weight-decay", type=float, default=1.0e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--init-weight", default="")
    train.add_argument("--conf", type=float, default=0.35)
    train.add_argument("--min-iou", type=float, default=0.5)
    train.add_argument("--max-det", type=int, default=20)
    train.add_argument("--obj-gain", type=float, default=1.0)
    train.add_argument("--cls-gain", type=float, default=1.0)
    train.add_argument("--box-gain", type=float, default=8.0)
    train.add_argument("--obj-pos-weight", type=float, default=40.0)
    train.add_argument("--closed-class-weight", type=float, default=4.0)
    train.add_argument("--closed-sample-weight", type=float, default=2.5)
    train.add_argument("--negative-sample-weight", type=float, default=1.0)
    train.add_argument("--obj-neg-weight", type=float, default=1.0)
    train.add_argument("--obj-focal-gamma", type=float, default=0.0)
    train.add_argument("--obj-hard-neg-topk", type=int, default=0)
    train.add_argument("--obj-hard-neg-weight", type=float, default=0.0)
    train.add_argument("--min-metric-positive-rate", type=float, default=0.0)
    train.add_argument("--min-metric-closed-rate", type=float, default=0.0)
    train.add_argument("--min-metric-precision", type=float, default=0.25)
    train.add_argument("--hard-negative-prob", type=float, default=0.35)
    train.add_argument("--positive-zoom-prob", type=float, default=0.0)
    train.add_argument("--grad-clip", type=float, default=5.0)
    train.add_argument("--export-onnx", action="store_true")
    train.add_argument("--opset", type=int, default=13)
    train.set_defaults(func=train_command)

    export = sub.add_parser("export", help="export a checkpoint to static ONNX")
    export.add_argument("--weight", required=True)
    export.add_argument("--onnx", required=True)
    export.add_argument("--opset", type=int, default=13)
    export.add_argument(
        "--raw",
        action="store_true",
        help="export only the 1x7x20x20 raw head; ESP-DL decode is done on device",
    )
    export.set_defaults(func=export_command)

    validate = sub.add_parser("validate", help="validate a checkpoint")
    validate.add_argument("--weight", required=True)
    validate.add_argument("--data", required=True)
    validate.add_argument("--split", default="test", choices=("train", "val", "test"))
    validate.add_argument("--batch", type=int, default=32)
    validate.add_argument("--workers", type=int, default=0)
    validate.add_argument("--device", default="auto")
    validate.add_argument("--conf", type=float, default=0.35)
    validate.add_argument("--min-iou", type=float, default=0.5)
    validate.add_argument("--max-det", type=int, default=20)
    validate.add_argument("--out-dir", default="runs/eye_state_direct/validation")
    validate.add_argument(
        "--qualitative-dir",
        help="optional unlabeled image dir to print detections for; not used for model selection",
    )
    validate.set_defaults(func=validate_command)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
