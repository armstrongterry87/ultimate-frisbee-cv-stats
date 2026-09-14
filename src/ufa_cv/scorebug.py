"""Score-bug detection on UFA broadcast footage.

The broadcast carries a persistent score bug along the bottom of the frame during live
play, and the director pulls it for replays, breaks, and pre/post-game. Detecting its
presence therefore yields a *live-play gate*: it answers "is this frame live action?"
without any inference on the play itself.

That matters for two reasons the audit called out. Replayed goals ingested as live play
double-count, which is the failure mode that destroys a coach's trust in a box score. And
gating cuts the frames a detector has to touch by ~40%.

Geometry below was measured on the 1920x1080 broadcast of game 2026-08-08-NY-BOS and is
specific to that season's graphics package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .video import Crop, extract_gray_strip

#: The bottom strip containing the score bug. Height is deliberately even: ffmpeg rounds an
#: odd crop height down and shifts the origin by a row, which silently misaligns every
#: downstream pixel constant in this module.
BUG_STRIP = Crop(x=535, y=914, width=860, height=84)

#: Rows of the strip occupied by the opaque graphic. Outside this band the strip shows
#: field, whose motion swamps any correlation.
GRAPHIC_ROWS = slice(23, 78)

#: Columns of the static left-hand league panel. It carries no scores or clock, so it is
#: constant whenever the bug is on screen, which makes it the cleanest presence signal.
PANEL_COLUMNS = slice(10, 150)

#: Presence threshold on the panel correlation. The score is strongly bimodal, so anything
#: in 0.3-0.7 gives the same answer to within 0.7% of frames.
PRESENT_THRESHOLD = 0.5


@dataclass(frozen=True)
class Segment:
    """A contiguous run of frames that are all live play, or all not."""

    start_s: float
    end_s: float
    live: bool

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def panel_correlation(strip: np.ndarray) -> np.ndarray:
    """Per-frame correlation of the league panel against its own reference.

    The reference is built from the frames whose panel has the highest variance, which are
    necessarily frames where the high-contrast graphic is present. This keeps the function
    self-contained rather than depending on a stored template that would have to be
    regenerated whenever the broadcaster changes its graphics package.
    """
    panel = np.asarray(
        strip[:, GRAPHIC_ROWS, PANEL_COLUMNS], dtype=np.float32
    ).reshape(len(strip), -1)
    panel -= panel.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(panel, axis=1) + 1e-6

    brightest = np.argsort(panel.var(axis=1))[-400:]
    reference = np.median(panel[brightest], axis=0)
    return (panel @ reference) / (norms * (np.linalg.norm(reference) + 1e-6))


def live_play_mask(strip: np.ndarray, threshold: float = PRESENT_THRESHOLD) -> np.ndarray:
    """Boolean mask over frames: True where the score bug is on screen."""
    return panel_correlation(strip) > threshold


def segments(mask: np.ndarray, sample_fps: float) -> list[Segment]:
    """Collapse a per-frame mask into contiguous live / not-live segments."""
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate(([0], edges, [len(mask)]))
    return [
        Segment(bounds[i] / sample_fps, bounds[i + 1] / sample_fps, bool(mask[bounds[i]]))
        for i in range(len(bounds) - 1)
    ]


def analyse(video: Path, cache: Path, sample_fps: float = 2.0) -> list[Segment]:
    """Decode ``video`` once and return its live-play segmentation."""
    strip = extract_gray_strip(video, BUG_STRIP, sample_fps, cache)
    return segments(live_play_mask(strip), sample_fps)
