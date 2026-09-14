"""Decoding helpers.

A full game is ~500k frames at 1080p, so anything that touches every frame has to avoid
per-frame Python work. These helpers push cropping, scaling and frame-rate sampling into a
single ffmpeg pass and hand back a memory-mapped array.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np


@dataclass(frozen=True)
class Crop:
    """A pixel rectangle in the source frame."""

    x: int
    y: int
    width: int
    height: int

    def as_ffmpeg(self) -> str:
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"


def extract_gray_strip(
    video: Path,
    crop: Crop,
    sample_fps: float,
    cache: Path,
    *,
    scale: tuple[int, int] | None = None,
    overwrite: bool = False,
) -> np.ndarray:
    """Decode ``video`` once, keeping a grayscale crop sampled at ``sample_fps``.

    Returns a memory-mapped ``(frames, height, width)`` array backed by ``cache``. Decoding
    a full game takes minutes, so an existing cache is reused unless ``overwrite`` is set.
    """
    out_h, out_w = (scale[1], scale[0]) if scale else (crop.height, crop.width)

    if overwrite or not cache.exists():
        filters = [f"fps={sample_fps}", crop.as_ffmpeg()]
        if scale:
            filters.append(f"scale={scale[0]}:{scale[1]}")
        filters.append("format=gray")
        cache.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-hide_banner", "-loglevel", "error",
                "-i", str(video),
                "-vf", ",".join(filters),
                "-f", "rawvideo", "-pix_fmt", "gray",
                "-y", str(cache),
            ],
            check=True,
        )

    return np.memmap(cache, dtype=np.uint8, mode="r").reshape(-1, out_h, out_w)


def probe_frame_count(video: Path) -> int:
    """Total frames, read from the container rather than by decoding."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
