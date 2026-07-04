#!/usr/bin/env python
"""Tkinter viewer for the helmet USB-OTG CDC camera debug port."""

from __future__ import annotations

import argparse
import base64
import json
import queue
import struct
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk


HELMET_USB_VID = 0x303A
HELMET_USB_PID = 0x4001
LEGACY_HEADER_STRUCT = struct.Struct("<4s8HIIi")
EXTENDED_HEADER_STRUCT = struct.Struct("<11H3I")
LEGACY_HEADER_SIZE = LEGACY_HEADER_STRUCT.size
EXTENDED_HEADER_SIZE = EXTENDED_HEADER_STRUCT.size

STATUS_FLAG_RUNNING = 1 << 0
STATUS_FLAG_FRAME_VALID = 1 << 1
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
        raise RuntimeError(f"invalid frame header: {raw_header.hex()}")

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


def request_snapshot(port, scale: int):
    port.reset_input_buffer()
    port.write(f"SNAP {scale}\n".encode("ascii"))
    port.flush()

    frame = read_frame_header(port)
    if frame["status"] != 0:
        raise RuntimeError(f"device returned error status=0x{frame['status'] & 0xFFFFFFFF:08x}")
    if frame["pixel_format"] != 1:
        raise RuntimeError(f"unsupported pixel format: {frame['pixel_format']}")
    expected = frame["width"] * frame["height"] * 2
    if frame["payload_size"] != expected:
        raise RuntimeError(f"bad payload size: got {frame['payload_size']}, expected {expected}")

    frame["payload"] = read_exact(port, frame["payload_size"])
    return frame


def rgb565_to_rgb_bytes(payload: bytes, width: int, height: int) -> bytes:
    out = bytearray(width * height * 3)
    dst = 0
    for off in range(0, len(payload), 2):
        pix = payload[off] | (payload[off + 1] << 8)
        out[dst] = ((pix >> 11) & 0x1F) * 255 // 31
        out[dst + 1] = ((pix >> 5) & 0x3F) * 255 // 63
        out[dst + 2] = (pix & 0x1F) * 255 // 31
        dst += 3
    return out


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


def frame_to_rgb_bytes(frame, overlay: bool = True) -> bytes:
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


def frame_to_ppm(frame, overlay: bool = True) -> bytes:
    width = frame["width"]
    height = frame["height"]
    rgb = frame_to_rgb_bytes(frame, overlay=overlay)
    return f"P6\n{width} {height}\n255\n".encode("ascii") + rgb


def frame_to_bmp(frame, overlay: bool = True) -> bytes:
    width = frame["width"]
    height = frame["height"]
    rgb = frame_to_rgb_bytes(frame, overlay=overlay)
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


class CameraWorker(threading.Thread):
    def __init__(self, port_name: str, scale: int, interval: float, out_queue: queue.Queue):
        super().__init__(daemon=True)
        self.port_name = port_name
        self.scale = scale
        self.interval = interval
        self.out_queue = out_queue
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def put_latest(self, item) -> None:
        try:
            self.out_queue.put_nowait(item)
            return
        except queue.Full:
            pass

        try:
            self.out_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.out_queue.put_nowait(item)
        except queue.Full:
            pass

    def run(self) -> None:
        serial, _ = import_serial()
        try:
            with serial.Serial(self.port_name, 921600, timeout=5.0, write_timeout=5.0) as port:
                port.reset_input_buffer()
                port.reset_output_buffer()
                while not self.stop_event.is_set():
                    start = time.monotonic()
                    frame = request_snapshot(port, self.scale)
                    elapsed = time.monotonic() - start
                    self.put_latest(("frame", (frame, elapsed)))
                    wait = max(0.0, self.interval - elapsed)
                    if self.stop_event.wait(wait):
                        break
        except Exception as exc:
            self.put_latest(("error", str(exc)))


class ViewerApp:
    def __init__(
        self,
        root: tk.Tk,
        default_port: str | None,
        default_scale: int,
        interval: float,
        capture_dir: Path,
    ):
        self.root = root
        self.root.title("Helmet Camera USB Viewer")
        self.serial, self.list_ports = import_serial()
        self.queue: queue.Queue = queue.Queue(maxsize=2)
        self.worker: CameraWorker | None = None
        self.photo = None
        self.latest_frame = None
        self.capture_dir = capture_dir
        self.ppm_fallback_path = Path(".codex_tmp") / "helmet_camera_view.ppm"
        self.interval = interval
        self.fps_start = time.monotonic()
        self.fps_count = 0
        self.fps = 0.0

        self.port_var = tk.StringVar(value=default_port or "")
        self.scale_var = tk.IntVar(value=default_scale)
        self.status_var = tk.StringVar(value="idle")
        self.capture_status_var = tk.StringVar(value=f"capture dir: {self.capture_dir.resolve()}")

        self._build_ui()
        self.root.bind("s", lambda _event: self.save_bmp())
        self.root.bind("o", lambda _event: self.capture_label("eye_open"))
        self.root.bind("c", lambda _event: self.capture_label("eye_closed"))
        self.root.bind("n", lambda _event: self.capture_label("no_eye"))
        self.refresh_ports(select_default=default_port)
        self.root.after(50, self.poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(toolbar, text="Port").pack(side=tk.LEFT)
        self.port_combo = ttk.Combobox(toolbar, textvariable=self.port_var, width=18, state="readonly")
        self.port_combo.pack(side=tk.LEFT, padx=(4, 10))

        ttk.Label(toolbar, text="Scale").pack(side=tk.LEFT)
        self.scale_combo = ttk.Combobox(
            toolbar,
            textvariable=self.scale_var,
            values=[1, 2, 3, 4, 5, 6, 7, 8],
            width=4,
            state="readonly",
        )
        self.scale_combo.pack(side=tk.LEFT, padx=(4, 10))

        ttk.Button(toolbar, text="Refresh", command=self.refresh_ports).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Start", command=self.start).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save BMP", command=self.save_bmp).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Open O", command=lambda: self.capture_label("eye_open")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Closed C", command=lambda: self.capture_label("eye_closed")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="No Eye N", command=lambda: self.capture_label("no_eye")).pack(side=tk.LEFT, padx=2)

        self.image_label = ttk.Label(self.root, anchor=tk.CENTER)
        self.image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        capture_status = ttk.Label(self.root, textvariable=self.capture_status_var, anchor=tk.W, padding=(8, 0, 8, 4))
        capture_status.pack(side=tk.BOTTOM, fill=tk.X)

        status = ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W, padding=8)
        status.pack(side=tk.BOTTOM, fill=tk.X)

    def refresh_ports(self, select_default: str | None = None) -> None:
        port_infos = list(self.list_ports.comports())
        ports = [port.device for port in port_infos]
        self.port_combo["values"] = ports
        if select_default and select_default in ports:
            self.port_var.set(select_default)
        else:
            helmet_port = next(
                (port.device for port in port_infos if port.vid == HELMET_USB_VID and port.pid == HELMET_USB_PID),
                None,
            )
            if helmet_port:
                self.port_var.set(helmet_port)
            elif not self.port_var.get() and ports:
                self.port_var.set(ports[0])
        details = [
            f"{port.device} {port.description} {port.hwid}"
            for port in port_infos
        ]
        self.status_var.set(f"ports: {' | '.join(details) if details else 'none'}")

    def start(self) -> None:
        if self.worker is not None:
            return
        port = self.port_var.get().strip()
        if not port:
            self.status_var.set("select a COM port")
            return
        scale = int(self.scale_var.get())
        self.worker = CameraWorker(port, scale, self.interval, self.queue)
        self.worker.start()
        self.status_var.set(f"running on {port}, scale={scale}")

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
            self.status_var.set("stopped")

    def save_bmp(self) -> None:
        if self.latest_frame is None:
            self.status_var.set("no frame to save")
            return
        out = Path("helmet_camera_view.bmp").resolve()
        out.write_bytes(frame_to_bmp(self.latest_frame))
        self.status_var.set(f"saved {out}")
        print(f"saved BMP: {out}", flush=True)

    def capture_label(self, label: str) -> None:
        if self.latest_frame is None:
            self.capture_status_var.set("no frame to capture")
            return

        if label not in {"eye_open", "eye_closed", "no_eye"}:
            self.capture_status_var.set(f"unknown capture label: {label}")
            return

        frame = self.latest_frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        seq = int(frame["frame_seq"])
        label_dir = self.capture_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        image_path = (label_dir / f"{label}_{timestamp}_f{seq}.bmp").resolve()
        meta_path = image_path.with_suffix(".json")
        image_path.write_bytes(frame_to_bmp(frame, overlay=False))

        flags = int(frame["flags"])
        metadata = {
            "label": label,
            "image_path": str(image_path),
            "saved_at": datetime.now().isoformat(timespec="milliseconds"),
            "frame_seq": seq,
            "width": int(frame["width"]),
            "height": int(frame["height"]),
            "src_width": int(frame["src_width"]),
            "src_height": int(frame["src_height"]),
            "scale": int(frame["scale"]),
            "model": "READY" if flags & STATUS_FLAG_MODEL_READY else "NO_MODEL",
            "eye_state": eye_state_name(int(frame["eye_state"])),
            "reason": reason_name(int(frame["reason"])),
            "eye_bbox": list(frame["eye_bbox"]),
            "eye_confidence": float(frame["eye_confidence"]),
            "eye_open_ratio": float(frame["eye_open_ratio"]),
            "perclos": float(frame["perclos"]),
            "blink_count": int(frame["blink_count"]),
            "no_eye_ms": int(frame["no_eye_ms"]),
            "last_infer_ms": int(frame["last_infer_ms"]),
        }
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        self.capture_status_var.set(f"captured {label}: {image_path}")
        print(f"captured {label}: {image_path}", flush=True)
        print(f"metadata: {meta_path}", flush=True)

    def display_frame(self, frame, grab_elapsed: float) -> None:
        ppm = frame_to_ppm(frame)
        try:
            encoded = base64.b64encode(ppm).decode("ascii")
            self.photo = tk.PhotoImage(data=encoded, format="PPM")
        except tk.TclError:
            self.ppm_fallback_path.parent.mkdir(parents=True, exist_ok=True)
            self.ppm_fallback_path.write_bytes(ppm)
            self.photo = tk.PhotoImage(file=str(self.ppm_fallback_path))

        self.image_label.configure(image=self.photo)
        self.latest_frame = frame
        self.fps_count += 1
        now = time.monotonic()
        fps_elapsed = now - self.fps_start
        if fps_elapsed >= 1.0:
            self.fps = self.fps_count / fps_elapsed
            self.fps_count = 0
            self.fps_start = now

        flags = frame["flags"]
        model = "READY" if flags & STATUS_FLAG_MODEL_READY else "NO_MODEL"
        eye = eye_state_name(frame["eye_state"])
        reason = reason_name(frame["reason"])
        bbox = frame["eye_bbox"]
        self.status_var.set(
            f"frame={frame['frame_seq']} "
            f"{frame['width']}x{frame['height']} "
            f"source={frame['src_width']}x{frame['src_height']} "
            f"scale={frame['scale']} fps={self.fps:.1f} grab={grab_elapsed * 1000:.0f}ms | "
            f"model={model} eye={eye} reason={reason} "
            f"conf={frame['eye_confidence']:.2f} open={frame['eye_open_ratio']:.2f} "
            f"perclos={frame['perclos']:.2f} blink={frame['blink_count']} "
            f"infer={frame['last_infer_ms']}ms noeye={frame['no_eye_ms']}ms "
            f"box={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        )

    def poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "frame":
                    frame, elapsed = payload
                    self.display_frame(frame, elapsed)
                elif kind == "error":
                    self.worker = None
                    self.status_var.set(f"error: {payload}")
        except queue.Empty:
            pass
        self.root.after(50, self.poll_queue)

    def close(self) -> None:
        self.stop()
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="USB-OTG CDC COM port, for example COM7")
    parser.add_argument("--scale", type=int, default=3, choices=range(1, 9), metavar="1..8")
    parser.add_argument("--interval", type=float, default=0.0, help="seconds between requested frames; 0 means as fast as possible")
    parser.add_argument(
        "--capture-dir",
        default="data/eye_state_capture",
        help="directory for labeled captures made with O/C/N keys",
    )
    args = parser.parse_args()

    root = tk.Tk()
    ViewerApp(root, args.port, args.scale, args.interval, Path(args.capture_dir))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
