#!/usr/bin/env python3
"""Train and export a direct eye-state YOLO detector.

The model classes are expected to be:

    0 eye_open
    1 eye_closed

Use this for the bootstrap detector trained from public/synthetic data. Final
accuracy still requires real helmet-camera samples.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch.nn as nn


def import_ultralytics():
    local_config = Path(".codex_tmp") / "eye_model"
    ultra_base = local_config / "ultralytics"
    mpl_base = local_config / "matplotlib"
    ultra_base.mkdir(parents=True, exist_ok=True)
    mpl_base.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(ultra_base.resolve())
    os.environ["MPLCONFIGDIR"] = str(mpl_base.resolve())
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "ultralytics is required for training/export.\n"
            f"Install it in this environment:\n  {sys.executable} -m pip install ultralytics"
        ) from exc
    return YOLO


def make_activation(name: str):
    if name == "silu":
        return None
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "hardswish":
        return nn.Hardswish(inplace=True)
    raise ValueError(f"unsupported activation: {name}")


def set_ultralytics_default_activation(name: str) -> None:
    activation = make_activation(name)
    if activation is None:
        return

    try:
        from ultralytics.nn.modules import Conv
    except Exception as exc:
        raise RuntimeError("failed to import ultralytics Conv module") from exc

    Conv.default_act = activation
    print(f"[info] ultralytics default activation={activation.__class__.__name__}")


def patch_model_activations(model, name: str) -> int:
    if name == "silu":
        return 0

    count = 0
    for module in model.modules():
        if hasattr(module, "act") and isinstance(module.act, nn.SiLU):
            module.act = make_activation(name)
            count += 1
    print(f"[info] patched_silu_activations={count} replacement={name}")
    return count


def train(args: argparse.Namespace) -> Path:
    YOLO = import_ultralytics()
    set_ultralytics_default_activation(args.activation)
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)

    model = YOLO(args.model)
    patch_model_activations(model.model, args.activation)
    results = model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained,
        patience=args.patience,
        cache=args.cache,
        seed=args.seed,
        exist_ok=args.exist_ok,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        mosaic=args.mosaic,
        scale=args.scale,
        erasing=args.erasing,
        box=args.box_gain,
        cls=args.cls_gain,
        dfl=args.dfl_gain,
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"training finished but best weight was not found: {best}")

    print(f"[info] best={best}")
    return best


def export_onnx(args: argparse.Namespace, weight: Path) -> Path:
    YOLO = import_ultralytics()
    model = YOLO(str(weight))
    patch_model_activations(model.model, args.activation)
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=False,
        half=False,
        int8=False,
    )
    onnx_path = Path(exported)
    print(f"[info] onnx={onnx_path}")
    return onnx_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="YOLO data.yaml")
    parser.add_argument("--weight", help="existing YOLO .pt to export without training")
    parser.add_argument("--model", default="yolov8n.yaml", help="YOLO model cfg/weights, e.g. yolov8n.yaml or yolov8n.pt")
    parser.add_argument(
        "--activation",
        default="silu",
        choices=("silu", "relu", "leaky_relu", "hardswish"),
        help="activation to use before training/export. Use relu/leaky_relu/hardswish for ESP-DL compatibility.",
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="runs/eye_state")
    parser.add_argument("--name", default="eye_state_detect_v0")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--pretrained", action="store_true", help="allow pretrained weights when the selected model supports them")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--erasing", type=float, default=0.4)
    parser.add_argument("--box-gain", type=float, default=7.5)
    parser.add_argument("--cls-gain", type=float, default=0.5)
    parser.add_argument("--dfl-gain", type=float, default=1.5)
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--simplify", action="store_true", help="run ONNX simplifier during export")
    args = parser.parse_args()

    if args.weight:
        best = Path(args.weight)
        if not best.exists():
            raise FileNotFoundError(best)
    else:
        if not args.data:
            raise SystemExit("--data is required when --weight is not provided")
        best = train(args)

    if args.export_onnx:
        export_onnx(args, best)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
