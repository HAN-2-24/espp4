#!/usr/bin/env python3
"""Split a YOLO-style ONNX final output into box and score graph outputs.

Ultralytics detect exports commonly end with:

    output0 = Concat(boxes, scores, axis=1)

Quantizing that final tensor as one output forces bbox coordinates and class
scores to share one scale. ESP-DL then may preserve boxes while destroying
0..1 confidence scores. This tool makes the two concat inputs graph outputs
directly so the exporter can quantize them separately.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import onnx
from onnx import TensorProto
from onnx import helper


def _value_infos(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    values = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        values[value.name] = value
    return values


def _copy_value_info(model: onnx.ModelProto, tensor_name: str, output_name: str) -> onnx.ValueInfoProto:
    infos = _value_infos(model)
    if tensor_name in infos:
        value = onnx.ValueInfoProto()
        value.CopyFrom(infos[tensor_name])
        value.name = output_name
        return value

    # Fallback for models where shape inference cannot prove the intermediate
    # shape. ESP-DL conversion will still validate the real tensor.
    return helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, None)


def _make_output_value_info(name: str, channels: int, candidates: int) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, channels, candidates])


def _shape_of(model: onnx.ModelProto, tensor_name: str) -> list[int] | None:
    info = _value_infos(model).get(tensor_name)
    if not info:
        return None
    shape = info.type.tensor_type.shape
    dims: list[int] = []
    for dim in shape.dim:
        if dim.dim_value <= 0:
            return None
        dims.append(int(dim.dim_value))
    return dims


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="source YOLO ONNX")
    parser.add_argument("--out", required=True, help="output ONNX")
    parser.add_argument("--output", default="", help="final YOLO output name; defaults to graph output 0")
    parser.add_argument("--boxes-source", default="", help="explicit tensor name to expose as boxes")
    parser.add_argument("--scores-source", default="", help="explicit tensor name to expose as scores")
    parser.add_argument("--raw-dfl", action="store_true", help="expose YOLOv8 raw DFL box logits instead of decoded xywh")
    parser.add_argument("--boxes-name", default="helmet_boxes")
    parser.add_argument("--scores-name", default="helmet_scores")
    args = parser.parse_args()

    src = Path(args.onnx)
    out = Path(args.out)
    model = onnx.load(str(src))
    inferred = onnx.shape_inference.infer_shapes(model)

    output_name = args.output or model.graph.output[0].name
    concat = None
    for node in model.graph.node:
        if output_name in node.output:
            concat = node
            break

    if concat is None:
        raise RuntimeError(f"could not find producer for graph output {output_name!r}")
    if concat.op_type != "Concat" or len(concat.input) < 2:
        raise RuntimeError(f"final producer is {concat.op_type}, expected Concat(boxes, scores)")

    boxes_input = args.boxes_source or concat.input[0]
    scores_input = args.scores_source or concat.input[1]

    if args.raw_dfl:
        if not args.boxes_source:
            boxes_input = "/model.22/Concat_output_0"
        if not args.scores_source:
            scores_input = "/model.22/Sigmoid_output_0"

    id_boxes = helper.make_node("Identity", [boxes_input], [args.boxes_name], name="helmet_boxes_identity")
    id_scores = helper.make_node("Identity", [scores_input], [args.scores_name], name="helmet_scores_identity")
    model.graph.node.extend([id_boxes, id_scores])

    inferred_values = _value_infos(inferred)
    for value_info in inferred.graph.value_info:
        if value_info.name not in _value_infos(model):
            model.graph.value_info.extend([value_info])

    if args.raw_dfl:
        boxes_shape = _shape_of(inferred, boxes_input)
        scores_shape = _shape_of(inferred, scores_input)
        candidates = scores_shape[2] if scores_shape and len(scores_shape) == 3 else 2100
        boxes_info = _make_output_value_info(args.boxes_name, 64, candidates)
        scores_info = _make_output_value_info(args.scores_name, 2, candidates)
    else:
        boxes_info = _copy_value_info(inferred, boxes_input, args.boxes_name)
        scores_info = _copy_value_info(inferred, scores_input, args.scores_name)

    del model.graph.output[:]
    model.graph.output.extend([boxes_info, scores_info])

    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    with tempfile.TemporaryDirectory() as tmpdir:
        unpruned = Path(tmpdir) / "with_debug_outputs.onnx"
        onnx.save(model, str(unpruned))
        onnx.utils.extract_model(
            str(unpruned),
            str(out),
            [value.name for value in model.graph.input],
            [args.boxes_name, args.scores_name],
        )

    print(f"[info] split {output_name}: boxes={boxes_input} -> {args.boxes_name}, scores={scores_input} -> {args.scores_name}")
    print(f"[info] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
