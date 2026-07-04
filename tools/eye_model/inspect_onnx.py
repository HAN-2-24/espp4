#!/usr/bin/env python3
"""Inspect ONNX input/output shapes before ESP-DL conversion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import onnx
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard.
    raise SystemExit(
        "onnx is required.\n"
        f"Install it in this environment:\n  {sys.executable} -m pip install onnx"
    ) from exc


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[str]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    dims: list[str] = []
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(str(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def collect_value_infos(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    values = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        values[value.name] = value
    return values


def known_rank(name: str, values: dict[str, onnx.ValueInfoProto]) -> int | None:
    value = values.get(name)
    if value is None:
        return None
    shape = tensor_shape(value)
    return len(shape) if shape else None


def report_negative_axes(model: onnx.ModelProto, values: dict[str, onnx.ValueInfoProto]) -> None:
    found = False
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == "axis" and attr.i < 0:
                found = True
                rank = next((known_rank(name, values) for name in node.input if known_rank(name, values)), None)
                print(f"[warn] negative axis node={node.name or node.op_type} axis={attr.i} rank={rank}")
            elif attr.name == "axes" and any(axis < 0 for axis in attr.ints):
                found = True
                rank = next((known_rank(name, values) for name in node.input if known_rank(name, values)), None)
                print(f"[warn] negative axes node={node.name or node.op_type} axes={list(attr.ints)} rank={rank}")
    if not found:
        print("[info] negative_axes=none")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", help="ONNX model path")
    parser.add_argument("--infer-shapes", action="store_true", help="run ONNX shape inference before printing")
    args = parser.parse_args()

    path = Path(args.onnx)
    if not path.exists():
        raise FileNotFoundError(path)

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    if args.infer_shapes:
        model = onnx.shape_inference.infer_shapes(model)

    opsets = ", ".join(f"{op.domain or 'ai.onnx'}:{op.version}" for op in model.opset_import)
    print(f"[info] model={path}")
    print(f"[info] ir_version={model.ir_version} opsets={opsets}")

    initializer_names = {init.name for init in model.graph.initializer}
    real_inputs = [value for value in model.graph.input if value.name not in initializer_names]

    print("[info] inputs:")
    for value in real_inputs:
        print(f"  {value.name}: {'x'.join(tensor_shape(value)) or '?'}")

    print("[info] outputs:")
    for value in model.graph.output:
        print(f"  {value.name}: {'x'.join(tensor_shape(value)) or '?'}")

    values = collect_value_infos(model)
    report_negative_axes(model, values)
    print(f"[info] nodes={len(model.graph.node)} initializers={len(model.graph.initializer)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
