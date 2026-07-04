#!/usr/bin/env python3
"""Manual bbox annotator for helmet eye-state capture images.

This tool only reads from the capture directory. It does not use or scan
`data/eye_state_negative`.

Controls:

- Mouse drag: draw/replace the eye box.
- `s`: save current annotation.
- `n` or Right: save and go next.
- `p` or Left: save and go previous.
- `c`: clear current box.
- `f`: set a full-frame box for near-eye patch images.
- `q` or Esc: save and quit.

Output:

- YOLO labels: `<capture-dir>/manual_labels/<label>/<image-stem>.txt`
- Visual checks: `<capture-dir>/visualized_manual/<label>/<image-stem>.jpg`
"""

from __future__ import annotations

import argparse
import base64
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard.
    raise SystemExit(
        "opencv-python and numpy are required.\n"
        f"Install them in this environment:\n  {sys.executable} -m pip install opencv-python numpy"
    ) from exc


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
CLASS_TO_ID = {"eye_open": 0, "eye_closed": 1}
LABELS = ("eye_open", "eye_closed", "no_eye")
PROHIBITED_PART = "eye_state_negative"


@dataclass
class Sample:
    path: Path
    label: str


def imread_color(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"empty image: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image


def imwrite_color(path: Path, image: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    encoded.tofile(str(path))


def list_samples(capture_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for label in LABELS:
        label_dir = capture_dir / label
        if not label_dir.exists():
            continue
        for path in sorted(p for p in label_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            samples.append(Sample(path=path, label=label))
    return samples


def label_path_for(capture_dir: Path, sample: Sample) -> Path:
    return capture_dir / "manual_labels" / sample.label / f"{sample.path.stem}.txt"


def visual_path_for(capture_dir: Path, sample: Sample) -> Path:
    return capture_dir / "visualized_manual" / sample.label / f"{sample.path.stem}.jpg"


def box_to_yolo(class_id: int, box: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return f"{class_id} {cx / width:.6f} {cy / height:.6f} {bw / width:.6f} {bh / height:.6f}\n"


def yolo_to_box(raw: str, width: int, height: int) -> tuple[int, int, int, int] | None:
    parts = raw.strip().split()
    if len(parts) != 5:
        return None
    _class_id, cx, cy, bw, bh = parts
    cx_f = float(cx) * width
    cy_f = float(cy) * height
    bw_f = float(bw) * width
    bh_f = float(bh) * height
    return (
        int(round(cx_f - bw_f / 2.0)),
        int(round(cy_f - bh_f / 2.0)),
        int(round(cx_f + bw_f / 2.0)),
        int(round(cy_f + bh_f / 2.0)),
    )


class AnnotatorApp:
    def __init__(self, root: tk.Tk, capture_dir: Path, samples: list[Sample], max_display: int):
        self.root = root
        self.capture_dir = capture_dir
        self.samples = samples
        self.max_display = max_display
        self.index = 0
        self.image_bgr: np.ndarray | None = None
        self.photo: tk.PhotoImage | None = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.box: tuple[int, int, int, int] | None = None
        self.drag_start: tuple[int, int] | None = None

        self.root.title("Helmet Capture Eye Annotator")
        self.status_var = tk.StringVar(value="")
        self._build_ui()
        self._bind_keys()
        self.load_sample(0)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text="Prev", command=self.prev_sample).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save", command=self.save_current).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Next", command=self.next_sample).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Clear", command=self.clear_box).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Full", command=self.full_box).pack(side=tk.LEFT, padx=2)

        self.canvas = tk.Canvas(self.root, bg="#202020", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        status = ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W, padding=6)
        status.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_keys(self) -> None:
        self.root.bind("s", lambda _event: self.save_current())
        self.root.bind("n", lambda _event: self.next_sample())
        self.root.bind("<Right>", lambda _event: self.next_sample())
        self.root.bind("p", lambda _event: self.prev_sample())
        self.root.bind("<Left>", lambda _event: self.prev_sample())
        self.root.bind("c", lambda _event: self.clear_box())
        self.root.bind("f", lambda _event: self.full_box())
        self.root.bind("q", lambda _event: self.quit())
        self.root.bind("<Escape>", lambda _event: self.quit())

    def current_sample(self) -> Sample:
        return self.samples[self.index]

    def load_sample(self, index: int) -> None:
        self.index = max(0, min(len(self.samples) - 1, index))
        sample = self.current_sample()
        self.image_bgr = imread_color(sample.path)
        self.box = self.read_existing_box(sample)
        self.redraw()

    def read_existing_box(self, sample: Sample) -> tuple[int, int, int, int] | None:
        if self.image_bgr is None:
            return None
        if sample.label == "no_eye":
            return None
        label_path = label_path_for(self.capture_dir, sample)
        if not label_path.exists():
            return None
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return None
        height, width = self.image_bgr.shape[:2]
        return yolo_to_box(lines[0], width, height)

    def image_to_display(self, x: int, y: int) -> tuple[int, int]:
        return int(round(x * self.scale + self.offset_x)), int(round(y * self.scale + self.offset_y))

    def display_to_image(self, x: int, y: int) -> tuple[int, int]:
        if self.image_bgr is None:
            return 0, 0
        height, width = self.image_bgr.shape[:2]
        ix = int(round((x - self.offset_x) / self.scale))
        iy = int(round((y - self.offset_y) / self.scale))
        return max(0, min(width - 1, ix)), max(0, min(height - 1, iy))

    def redraw(self) -> None:
        assert self.image_bgr is not None
        sample = self.current_sample()
        rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        self.scale = min(self.max_display / width, self.max_display / height, 1.0)
        disp_w = max(1, int(round(width * self.scale)))
        disp_h = max(1, int(round(height * self.scale)))
        resized = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        ppm = b"P6\n%d %d\n255\n" % (disp_w, disp_h) + resized.tobytes()
        encoded = base64.b64encode(ppm).decode("ascii")
        self.photo = tk.PhotoImage(data=encoded, format="PPM")
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.offset_x = 0
        self.offset_y = 0
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.photo)

        if self.box is not None:
            x1, y1 = self.image_to_display(self.box[0], self.box[1])
            x2, y2 = self.image_to_display(self.box[2], self.box[3])
            color = "#40ff80" if sample.label == "eye_open" else "#ff4040"
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)

        label_path = label_path_for(self.capture_dir, sample)
        visual_path = visual_path_for(self.capture_dir, sample)
        self.status_var.set(
            f"{self.index + 1}/{len(self.samples)} {sample.label} {sample.path} | "
            f"label={label_path} visual={visual_path}"
        )

    def on_mouse_down(self, event) -> None:
        self.drag_start = self.display_to_image(event.x, event.y)

    def on_mouse_drag(self, event) -> None:
        if self.drag_start is None:
            return
        end = self.display_to_image(event.x, event.y)
        self.box = (self.drag_start[0], self.drag_start[1], end[0], end[1])
        self.redraw()

    def on_mouse_up(self, event) -> None:
        if self.drag_start is None:
            return
        end = self.display_to_image(event.x, event.y)
        self.box = normalize_box((self.drag_start[0], self.drag_start[1], end[0], end[1]))
        self.drag_start = None
        self.redraw()

    def clear_box(self) -> None:
        self.box = None
        self.redraw()

    def full_box(self) -> None:
        assert self.image_bgr is not None
        height, width = self.image_bgr.shape[:2]
        margin_x = int(width * 0.04)
        margin_y = int(height * 0.04)
        self.box = (margin_x, margin_y, width - margin_x - 1, height - margin_y - 1)
        self.redraw()

    def save_current(self) -> None:
        assert self.image_bgr is not None
        sample = self.current_sample()
        label_path = label_path_for(self.capture_dir, sample)
        label_path.parent.mkdir(parents=True, exist_ok=True)

        if sample.label == "no_eye" or self.box is None:
            label_path.write_text("", encoding="utf-8")
        else:
            height, width = self.image_bgr.shape[:2]
            class_id = CLASS_TO_ID[sample.label]
            label_path.write_text(box_to_yolo(class_id, normalize_box(self.box), width, height), encoding="utf-8")

        self.write_visual(sample)
        self.status_var.set(f"saved {label_path}")

    def write_visual(self, sample: Sample) -> None:
        assert self.image_bgr is not None
        visual = self.image_bgr.copy()
        if sample.label != "no_eye" and self.box is not None:
            x1, y1, x2, y2 = normalize_box(self.box)
            color = (64, 255, 128) if sample.label == "eye_open" else (64, 64, 255)
            cv2.rectangle(visual, (x1, y1), (x2, y2), color, 2)
        else:
            color = (255, 208, 64)
        cv2.putText(visual, sample.label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        imwrite_color(visual_path_for(self.capture_dir, sample), visual)

    def next_sample(self) -> None:
        self.save_current()
        self.load_sample(self.index + 1)

    def prev_sample(self) -> None:
        self.save_current()
        self.load_sample(self.index - 1)

    def quit(self) -> None:
        self.save_current()
        self.root.destroy()


def normalize_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", default="data/eye_state_capture")
    parser.add_argument("--max-display", type=int, default=900)
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir)
    if PROHIBITED_PART in {part.lower() for part in capture_dir.parts}:
        raise SystemExit(f"refusing prohibited source path: {capture_dir}")
    samples = list_samples(capture_dir)
    if not samples:
        raise SystemExit(f"no capture images found under {capture_dir}")

    root = tk.Tk()
    AnnotatorApp(root, capture_dir, samples, args.max_display)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
