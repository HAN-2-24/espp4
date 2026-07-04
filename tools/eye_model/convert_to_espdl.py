#!/usr/bin/env python3
"""Convert an ONNX eye model to an ESP-DL model for ESP32-P4.

The script keeps the source ONNX untouched. It writes a temporary working ONNX
next to the requested .espdl output, normalizes negative ONNX axes when shape
inference can prove the tensor rank, then calls esp-ppq.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import onnx
import torch
from torch.utils.data import DataLoader

try:
    import cv2
except Exception:  # pragma: no cover - optional unless --calib-dir is used.
    cv2 = None

from esp_ppq.api import QuantizationSettingFactory, espdl_quantize_onnx
from esp_ppq.core import QuantizationProperty, TargetPlatform
from esp_ppq.lib import register_network_quantizer
from esp_ppq.quantization.quantizer.EspdlQuantizer import EspdlQuantizer


def parse_size(text: str) -> Optional[int]:
    value = text.strip()
    if not value:
        return None
    upper = value.upper()
    mul = 1
    if upper.endswith("K"):
        mul = 1024
        upper = upper[:-1]
    elif upper.endswith("M"):
        mul = 1024 * 1024
        upper = upper[:-1]
    try:
        return int(upper, 0) * mul
    except ValueError:
        return None


def find_partition_size(partitions_csv: Path, label: str) -> Optional[int]:
    if not partitions_csv.exists():
        return None
    with partitions_csv.open("r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            if row[0].strip() == label and len(row) >= 5:
                return parse_size(row[4])
    return None


def value_info_shape(value_info: onnx.ValueInfoProto) -> Optional[list[int]]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims: list[int] = []
    for dim in tensor_type.shape.dim:
        if dim.dim_value <= 0:
            return None
        dims.append(int(dim.dim_value))
    return dims


def collect_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception as exc:
        print(f"[warn] ONNX shape inference failed: {exc}")
        inferred = model

    shapes: dict[str, list[int]] = {}
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    for value in values:
        shape = value_info_shape(value)
        if shape:
            shapes[value.name] = shape
    return shapes


def first_known_input_rank(node: onnx.NodeProto, shapes: dict[str, list[int]]) -> Optional[int]:
    for name in node.input:
        shape = shapes.get(name)
        if shape:
            return len(shape)
    return None


def normalize_negative_axes(model: onnx.ModelProto) -> tuple[int, list[str]]:
    """Rewrite legal ONNX negative axes to positive axes for ESP-DL exporter."""
    shapes = collect_shapes(model)
    fixed = 0
    skipped: list[str] = []

    for node in model.graph.node:
        rank = None
        for attr in node.attribute:
            if attr.name not in ("axis", "axes"):
                continue
            if rank is None:
                rank = first_known_input_rank(node, shapes)
            if rank is None:
                skipped.append(f"{node.name or node.op_type}: unknown rank for {attr.name}")
                continue

            if attr.name == "axis" and attr.i < 0:
                attr.i = int(attr.i) + rank
                fixed += 1
            elif attr.name == "axes":
                axes = list(attr.ints)
                if any(axis < 0 for axis in axes):
                    del attr.ints[:]
                    attr.ints.extend([(int(axis) + rank) if axis < 0 else int(axis) for axis in axes])
                    fixed += 1

    return fixed, skipped


def get_single_input_shape(model: onnx.ModelProto, override: Optional[str]) -> list[int]:
    if override:
        shape = [int(part.strip()) for part in override.replace("x", ",").split(",") if part.strip()]
        if not shape:
            raise ValueError("--input-shape is empty")
        return shape

    real_inputs = []
    initializer_names = {init.name for init in model.graph.initializer}
    for value in model.graph.input:
        if value.name not in initializer_names:
            real_inputs.append(value)
    if len(real_inputs) != 1:
        raise ValueError(f"expected one real ONNX input, got {len(real_inputs)}; pass --input-shape")

    shape = value_info_shape(real_inputs[0])
    if not shape:
        raise ValueError("ONNX input has dynamic/unknown shape; pass --input-shape")
    return shape


def image_layout_for_shape(shape: list[int], requested: str) -> str:
    if requested != "auto":
        return requested
    if len(shape) == 4 and shape[1] == 3:
        return "nchw"
    if len(shape) == 4 and shape[3] == 3:
        return "nhwc"
    raise ValueError("cannot infer image layout from input shape; pass --image-layout")


def load_calibration_images(
    calib_dir: Path,
    shape: list[int],
    steps: int,
    layout: str,
    scale: float,
) -> list[torch.Tensor]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required when --calib-dir is used")
    paths = sorted(
        p
        for p in calib_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not paths:
        raise ValueError(f"no calibration images found in {calib_dir}")

    if layout == "nchw":
        _, channels, height, width = shape
    else:
        _, height, width, channels = shape
    if channels != 3:
        raise ValueError(f"image calibration expects 3 channels, got shape {shape}")

    tensors: list[torch.Tensor] = []
    for idx in range(steps):
        path = paths[idx % len(paths)]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read calibration image {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        arr = image.astype(np.float32) * scale
        if layout == "nchw":
            arr = np.transpose(arr, (2, 0, 1))[None, ...]
        else:
            arr = arr[None, ...]
        tensors.append(torch.from_numpy(arr))
    return tensors


def make_calibration_samples(args: argparse.Namespace, shape: list[int]) -> list[torch.Tensor]:
    if args.calib_dir:
        layout = image_layout_for_shape(shape, args.image_layout)
        return load_calibration_images(Path(args.calib_dir), shape, args.calib_steps, layout, args.input_scale)

    print("[warn] using random calibration samples; output is for conversion smoke-test only")
    return [torch.rand(*shape, dtype=torch.float32) for _ in range(args.calib_steps)]


class P4PerTensorEspdlQuantizer(EspdlQuantizer):
    """ESP-DL INT8 quantizer variant that avoids P4 per-channel weights.

    The local ESP-DL runtime on this project rejected the exported model with
    "Currently only support PER_TENSOR quantization". This class is an explicit
    compatibility experiment, not a model-quality fix.
    """

    def create_espdl_quant_config(self, operation, num_of_bits, quant_min, quant_max, bias_bits):
        config = super().create_espdl_quant_config(operation, num_of_bits, quant_min, quant_max, bias_bits)
        if operation.type not in {"Conv", "ConvTranspose", "Gemm"}:
            return config

        for tensor_config in config.input_quantization_config:
            if tensor_config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                tensor_config.policy = self.quantize_policy
        return config


def configure_esp_ppq_runtime(args: argparse.Namespace) -> None:
    # esp_ppq.api.interface imports FORMATTER_FUSE_SWISH by value, so update
    # both modules before espdl_quantize_onnx calls load_onnx_graph().
    import esp_ppq.api.interface as ppq_interface
    import esp_ppq.core.common as ppq_common

    fuse_swish = bool(args.enable_swish_fusion)
    ppq_common.FORMATTER_FUSE_SWISH = fuse_swish
    ppq_interface.FORMATTER_FUSE_SWISH = fuse_swish
    print(f"[info] ppq_formatter_fuse_swish={fuse_swish}")

    if args.force_p4_per_tensor:
        register_network_quantizer(P4PerTensorEspdlQuantizer, TargetPlatform.ESPDL_INT8)
        print("[warn] registered experimental ESP32-P4 INT8 per-tensor quantizer")


def iter_outputs(base: Path) -> Iterable[Path]:
    stem = base.with_suffix("")
    for suffix in (".espdl", ".info", ".json"):
        path = stem.with_suffix(suffix)
        if path.exists():
            yield path


def inspect_exported_model(base: Path) -> tuple[int, int]:
    stem = base.with_suffix("")
    json_path = stem.with_suffix(".json")
    info_path = stem.with_suffix(".info")

    per_channel_count = 0
    swish_count = 0

    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        configs = data.get("configs", {})
        for op_configs in configs.values():
            if not isinstance(op_configs, dict):
                continue
            for tensor_config in op_configs.values():
                if not isinstance(tensor_config, dict):
                    continue
                policy = tensor_config.get("policy", {})
                if policy.get("PER_CHANNEL"):
                    per_channel_count += 1

    if info_path.exists():
        text = info_path.read_text(encoding="utf-8", errors="ignore")
        swish_count = text.count("Swish")

    if per_channel_count:
        print(
            "[warn] exported quant config contains "
            f"{per_channel_count} PER_CHANNEL tensors; this ESP-DL runtime has already failed to load PER_CHANNEL."
        )
    if swish_count:
        print(
            "[warn] exported model info contains "
            f"{swish_count} Swish references; this ESP-DL runtime has already failed to load Swish."
        )

    return per_channel_count, swish_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="source ONNX file")
    parser.add_argument("--out", required=True, help="output .espdl file")
    parser.add_argument("--target", default="esp32p4", help="ESP-DL target chip")
    parser.add_argument("--bits", type=int, default=8, choices=(8, 16), help="quantization bits")
    parser.add_argument("--calib-steps", type=int, default=4, help="number of calibration samples")
    parser.add_argument("--calib-dir", help="optional directory of calibration images")
    parser.add_argument("--input-shape", help="override ONNX input shape, e.g. 1,3,640,640")
    parser.add_argument("--image-layout", default="auto", choices=("auto", "nchw", "nhwc"))
    parser.add_argument("--input-scale", type=float, default=1.0 / 255.0)
    parser.add_argument("--no-axis-fix", action="store_true", help="do not normalize negative axes")
    parser.add_argument(
        "--enable-fusion",
        action="store_true",
        help="allow ESP-PPQ graph fusion. Disabled by default because YOLO SiLU can fuse to unsupported Swish.",
    )
    parser.add_argument(
        "--enable-swish-fusion",
        action="store_true",
        help="allow ESP-PPQ graph-format Swish fusion. Disabled by default because this runtime rejected Swish.",
    )
    parser.add_argument(
        "--force-p4-per-tensor",
        action="store_true",
        help="experimental: replace the ESP32-P4 INT8 quantizer so Conv/Gemm weights and bias stay PER_TENSOR.",
    )
    parser.add_argument(
        "--fail-on-runtime-unsupported",
        action="store_true",
        help="return an error if exported config/info still contains board-rejected PER_CHANNEL or Swish.",
    )
    parser.add_argument("--error-report", action="store_true", help="run esp-ppq error reports")
    parser.add_argument("--partition-csv", default="partitions.csv", help="partition table for size check")
    parser.add_argument("--partition-label", default="eye_model", help="partition label used for model size check")
    args = parser.parse_args()

    src = Path(args.onnx)
    out = Path(args.out)
    if out.suffix.lower() != ".espdl":
        raise ValueError("--out must end with .espdl")
    if not src.exists():
        raise FileNotFoundError(src)

    out.parent.mkdir(parents=True, exist_ok=True)
    work_onnx = out.with_suffix(".work.onnx")
    shutil.copyfile(src, work_onnx)

    model = onnx.load(str(work_onnx))
    input_shape = get_single_input_shape(model, args.input_shape)
    print(f"[info] input_shape={input_shape}")

    if not args.no_axis_fix:
        fixed, skipped = normalize_negative_axes(model)
        if fixed:
            onnx.save(model, str(work_onnx))
        print(f"[info] normalized_negative_axes={fixed}")
        for item in skipped:
            print(f"[warn] skipped axis normalization: {item}")

    onnx.checker.check_model(onnx.load(str(work_onnx)))

    samples = make_calibration_samples(args, input_shape)
    loader = DataLoader(samples, batch_size=None, shuffle=False)

    configure_esp_ppq_runtime(args)

    setting = QuantizationSettingFactory.espdl_setting()
    setting.fusion = bool(args.enable_fusion)
    setting.graph_format_setting.fuse_swish = bool(args.enable_swish_fusion)

    graph = espdl_quantize_onnx(
        onnx_import_file=str(work_onnx),
        espdl_export_file=str(out),
        calib_dataloader=loader,
        calib_steps=args.calib_steps,
        input_shape=input_shape,
        target=args.target,
        num_of_bits=args.bits,
        setting=setting,
        device="cpu",
        error_report=args.error_report,
        export_config=True,
        export_test_values=False,
        verbose=1,
    )

    print(f"[info] converted_ops={len(graph.operations)} variables={len(graph.variables)}")
    for path in iter_outputs(out):
        print(f"[info] output {path} {path.stat().st_size} bytes")

    per_channel_count, swish_count = inspect_exported_model(out)
    if args.fail_on_runtime_unsupported and (per_channel_count or swish_count):
        raise RuntimeError(
            "exported ESP-DL model still contains runtime-unsupported "
            f"features: PER_CHANNEL={per_channel_count}, Swish={swish_count}"
        )

    partition_size = find_partition_size(Path(args.partition_csv), args.partition_label)
    if partition_size is not None and out.exists():
        model_size = out.stat().st_size
        status = "fits" if model_size <= partition_size else "too_large"
        print(
            f"[info] {args.partition_label}_partition={partition_size} bytes "
            f"espdl={model_size} bytes status={status}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
