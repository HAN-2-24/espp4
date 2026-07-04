#!/usr/bin/env python
"""Grab one RGB565 camera snapshot from the helmet USB-OTG CDC debug port."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


HELMET_USB_VID = 0x303A
HELMET_USB_PID = 0x4001
LEGACY_HEADER_STRUCT = struct.Struct("<4s8HIIi")
EXTENDED_HEADER_STRUCT = struct.Struct("<11H3I")
LEGACY_HEADER_SIZE = LEGACY_HEADER_STRUCT.size
EXTENDED_HEADER_SIZE = EXTENDED_HEADER_STRUCT.size

STATUS_FLAG_MODEL_READY = 1 << 2
STATUS_FLAG_EYE_VALID = 1 << 3
STATUS_FLAG_EYE_CLOSED = 1 << 4


def import_serial():
    try:
        import serial
        from serial.tools import list_ports
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "pyserial is not installed in this Python environment.\n"
            "Install it with:\n"
            f"  {sys.executable} -m pip install pyserial"
        ) from exc
    return serial, list_ports


def read_exact(port, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = port.read(remaining)
        if not chunk:
            raise TimeoutError(f"timeout while reading {size} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame_header(port) -> dict:
    raw_header = read_exact(port, LEGACY_HEADER_SIZE)
    (
        magic,
        version,
        header_size,
        width,
        height,
        src_width,
        src_height,
        actual_scale,
        pixel_format,
        frame_seq,
        payload_size,
        status,
    ) = LEGACY_HEADER_STRUCT.unpack(raw_header)

    if magic != b"HVC1" or version < 1 or header_size < LEGACY_HEADER_SIZE:
        raise SystemExit(f"invalid frame header: {raw_header.hex()}")

    extra = b""
    if header_size > LEGACY_HEADER_SIZE:
        extra = read_exact(port, header_size - LEGACY_HEADER_SIZE)

    header = {
        "version": version,
        "header_size": header_size,
        "width": width,
        "height": height,
        "src_width": src_width,
        "src_height": src_height,
        "scale": actual_scale,
        "pixel_format": pixel_format,
        "frame_seq": frame_seq,
        "payload_size": payload_size,
        "status": status,
        "flags": 0,
        "eye_state": 0,
        "reason": 0,
        "eye_bbox": (0, 0, 0, 0),
        "eye_confidence": 0.0,
        "eye_open_ratio": 0.0,
        "perclos": 0.0,
        "blink_count": 0,
        "no_eye_ms": 0,
        "last_infer_ms": 0,
    }

    if version >= 2 and len(extra) >= EXTENDED_HEADER_SIZE:
        (
            flags,
            eye_state,
            reason,
            bbox_x,
            bbox_y,
            bbox_w,
            bbox_h,
            eye_confidence_milli,
            eye_open_milli,
            perclos_milli,
            _reserved0,
            blink_count,
            no_eye_ms,
            last_infer_ms,
        ) = EXTENDED_HEADER_STRUCT.unpack(extra[:EXTENDED_HEADER_SIZE])
        header.update(
            {
                "flags": flags,
                "eye_state": eye_state,
                "reason": reason,
                "eye_bbox": (bbox_x, bbox_y, bbox_w, bbox_h),
                "eye_confidence": eye_confidence_milli / 1000.0,
                "eye_open_ratio": eye_open_milli / 1000.0,
                "perclos": perclos_milli / 1000.0,
                "blink_count": blink_count,
                "no_eye_ms": no_eye_ms,
                "last_infer_ms": last_infer_ms,
            }
        )

    return header


def eye_state_name(state: int) -> str:
    return {
        0: "UNKNOWN",
        1: "OPEN",
        2: "CLOSED",
        3: "INVALID",
        4: "NO_MODEL",
    }.get(state, f"STATE_{state}")


def reason_name(reason: int) -> str:
    return {
        0: "OK",
        1: "NO_MODEL",
        2: "NO_DETECTION",
        3: "BUSY",
        4: "LOW_CONF",
        5: "CAMERA_ERROR",
    }.get(reason, f"REASON_{reason}")


def rgb565_to_rgb_bytes(payload: bytes, width: int, height: int) -> bytearray:
    out = bytearray(width * height * 3)
    dst = 0
    for off in range(0, len(payload), 2):
        pix = payload[off] | (payload[off + 1] << 8)
        out[dst] = ((pix >> 11) & 0x1F) * 255 // 31
        out[dst + 1] = ((pix >> 5) & 0x3F) * 255 // 63
        out[dst + 2] = (pix & 0x1F) * 255 // 31
        dst += 3
    return out


def draw_rect_rgb(rgb: bytearray, width: int, height: int, bbox, color: tuple[int, int, int], thickness: int) -> None:
    x, y, w, h = bbox
    if width <= 0 or height <= 0 or w <= 0 or h <= 0:
        return
    x1 = max(0, min(width - 1, x))
    y1 = max(0, min(height - 1, y))
    x2 = max(0, min(width - 1, x + w - 1))
    y2 = max(0, min(height - 1, y + h - 1))
    if x2 <= x1 or y2 <= y1:
        return

    r, g, b = color
    for t in range(max(1, thickness)):
        top = y1 + t
        bottom = y2 - t
        left = x1 + t
        right = x2 - t
        if top > bottom or left > right:
            break
        for px in range(left, right + 1):
            for py in (top, bottom):
                off = (py * width + px) * 3
                rgb[off:off + 3] = bytes((r, g, b))
        for py in range(top, bottom + 1):
            for px in (left, right):
                off = (py * width + px) * 3
                rgb[off:off + 3] = bytes((r, g, b))


def frame_to_rgb_bytes(frame: dict, overlay: bool) -> bytes:
    width = frame["width"]
    height = frame["height"]
    rgb = rgb565_to_rgb_bytes(frame["payload"], width, height)
    if overlay:
        bbox_x, bbox_y, bbox_w, bbox_h = frame["eye_bbox"]
        if bbox_w > 0 and bbox_h > 0:
            scale = max(1, int(frame["scale"]))
            scaled_bbox = (
                bbox_x // scale,
                bbox_y // scale,
                max(1, (bbox_w + scale - 1) // scale),
                max(1, (bbox_h + scale - 1) // scale),
            )
            flags = frame["flags"]
            if flags & STATUS_FLAG_EYE_VALID:
                color = (255, 64, 64) if flags & STATUS_FLAG_EYE_CLOSED else (64, 255, 128)
            else:
                color = (255, 208, 64)
            draw_rect_rgb(rgb, width, height, scaled_bbox, color, 2)
    return bytes(rgb)


def rgb_to_bmp(rgb: bytes, width: int, height: int) -> bytes:
    row_stride = (width * 3 + 3) & ~3
    image_size = row_stride * height
    file_size = 54 + image_size

    header = bytearray(54)
    header[0:2] = b"BM"
    struct.pack_into("<I", header, 2, file_size)
    struct.pack_into("<I", header, 10, 54)
    struct.pack_into("<I", header, 14, 40)
    struct.pack_into("<i", header, 18, width)
    struct.pack_into("<i", header, 22, height)
    struct.pack_into("<H", header, 26, 1)
    struct.pack_into("<H", header, 28, 24)
    struct.pack_into("<I", header, 34, image_size)

    out = bytearray(header)
    pad = b"\x00" * (row_stride - width * 3)

    for y in range(height - 1, -1, -1):
        row_start = y * width * 3
        for x in range(width):
            off = row_start + x * 3
            r, g, b = rgb[off], rgb[off + 1], rgb[off + 2]
            out.extend((b, g, r))
        out.extend(pad)

    return bytes(out)


def list_serial_ports() -> None:
    _, list_ports = import_serial()
    for port in list_ports.comports():
        print(f"{port.device}\t{port.description}\t{port.hwid}")


def find_helmet_port() -> str | None:
    _, list_ports = import_serial()
    for port in list_ports.comports():
        if port.vid == HELMET_USB_VID and port.pid == HELMET_USB_PID:
            return port.device
    return None


def grab_snapshot(args) -> None:
    serial, _ = import_serial()
    out_path = Path(args.out)
    port_name = args.port or find_helmet_port()
    if not port_name:
        raise SystemExit("helmet USB camera port not found; plug the USB-OTG port and run --list")

    with serial.Serial(port_name, args.baud, timeout=args.timeout, write_timeout=args.timeout) as port:
        port.reset_input_buffer()
        port.reset_output_buffer()
        port.write(f"SNAP {args.scale}\n".encode("ascii"))
        port.flush()

        frame = read_frame_header(port)
        if frame["status"] != 0:
            raise SystemExit(f"device returned error status=0x{frame['status'] & 0xFFFFFFFF:08x}")
        if frame["pixel_format"] != 1:
            raise SystemExit(f"unsupported pixel format: {frame['pixel_format']}")
        expected = frame["width"] * frame["height"] * 2
        if frame["payload_size"] != expected:
            raise SystemExit(f"bad payload size: got {frame['payload_size']}, expected {expected}")

        frame["payload"] = read_exact(port, frame["payload_size"])

    rgb = frame_to_rgb_bytes(frame, overlay=not args.no_overlay)
    out_path.write_bytes(rgb_to_bmp(rgb, frame["width"], frame["height"]))
    model = "READY" if frame["flags"] & STATUS_FLAG_MODEL_READY else "NO_MODEL"
    bbox = frame["eye_bbox"]
    print(
        f"saved {out_path} "
        f"frame={frame['frame_seq']} size={frame['width']}x{frame['height']} "
        f"source={frame['src_width']}x{frame['src_height']} scale={frame['scale']} "
        f"model={model} eye={eye_state_name(frame['eye_state'])} reason={reason_name(frame['reason'])} "
        f"conf={frame['eye_confidence']:.2f} open={frame['eye_open_ratio']:.2f} "
        f"perclos={frame['perclos']:.2f} blink={frame['blink_count']} "
        f"infer={frame['last_infer_ms']}ms noeye={frame['no_eye_ms']}ms "
        f"box={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list available serial ports")
    parser.add_argument("--port", help="USB-OTG CDC COM port, for example COM7; auto-detects VID:PID 303A:4001 when omitted")
    parser.add_argument("--baud", type=int, default=921600, help="ignored by USB CDC but required by pyserial")
    parser.add_argument("--timeout", type=float, default=10.0, help="read/write timeout in seconds")
    parser.add_argument("--scale", type=int, default=4, choices=range(1, 9), metavar="1..8")
    parser.add_argument("--out", default="helmet_camera.bmp", help="output BMP path")
    parser.add_argument("--no-overlay", action="store_true", help="save the raw frame without drawing the eye box")
    args = parser.parse_args()

    if args.list:
        list_serial_ports()
        return 0
    grab_snapshot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
