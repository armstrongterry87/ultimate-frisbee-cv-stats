"""Read the broadcast scoreboard to recover the scoring timeline.

Why this exists: the API gives throw-level events in wall-clock time, the video has its own
timeline, and quarter breaks alone align the two only to about +/-30 s -- far too coarse to
label a throw that lasts a few seconds. Each goal changes the on-screen score, so the score
changes are dense alignment anchors, roughly one every three minutes.

**This is a calibration tool, not a general scoreboard reader.** It is handed the game's
final score and uses it to self-calibrate, so it aligns footage of a game whose stats we
already have. It cannot tell you the score of a game the API does not cover.

Approach, and why each part is needed:

- Digits are segmented as connected components and split by **hole count** before
  clustering. Clustering raw bitmaps confuses 7 with 8; topology cannot, since 7 encloses
  no region and 8 encloses two. This single feature removed the dominant error.
- The digit-to-glyph mapping is derived from **long plateaus only**. Ordering states by
  first appearance fails, because the graphic's fade-in produces garbage reads *before* the
  real 0-0 and poisons the ordering.
- Impossible readings are rejected by range, then the timeline is chosen as the **longest
  chain** in which every step advances exactly one side by exactly one. A stateful
  accept/reject filter deadlocks permanently after a single missed step; the chain search
  degrades gracefully instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: Rows of the score-bug strip holding the score digits.
DIGIT_ROWS = slice(30, 72)

#: Columns of each side's score. The away number is right-aligned against the divider at
#: x~425 and the home number left-aligned after it; "NY"/"BOS" text starts beyond x~507.
SCORE_COLUMNS = {"away": slice(356, 424), "home": slice(431, 496)}

#: Digits render near-white over a translucent panel, so a high threshold isolates them
#: and removes the panel's constant jitter over the moving field behind it.
INK_THRESHOLD = 200

_GLYPH_SIZE = (20, 28)

#: Digits grouped by number of enclosed regions: {1,2,3,5,7} | {0,4,6,9} | {8}.
_HOLE_GROUPS = {0: 5, 1: 4, 2: 1}


@dataclass(frozen=True)
class ScoreState:
    """A period during which the scoreboard showed one score."""

    away: int
    home: int
    start_s: float
    end_s: float


def _hole_count(glyph: np.ndarray) -> int:
    padded = np.pad(glyph, 1, constant_values=0)
    count, _ = cv2.connectedComponents((padded == 0).astype(np.uint8), connectivity=4)
    return count - 2          # drop the outside region and the zero label


def _segment_digits(strip_frame: np.ndarray, columns: slice):
    """Digit-sized blobs in one score window, left to right, or None if unreadable."""
    window = np.asarray(strip_frame[DIGIT_ROWS, columns], dtype=np.uint8)
    binary = (window > INK_THRESHOLD).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    found = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if h >= 24 and w >= 5 and area >= 100:
            mask = (labels[y : y + h, x : x + w] == label).astype(np.uint8)
            resized = cv2.resize(
                mask.astype(np.float32), _GLYPH_SIZE, interpolation=cv2.INTER_LINEAR
            )
            found.append((x, _hole_count(mask), resized.ravel()))
    found.sort(key=lambda item: item[0])
    return found if 1 <= len(found) <= 2 else None


def _cluster_glyphs(records: list) -> dict:
    """Assign every glyph a cluster id, clustering within each hole-count group."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    assignment = {}
    for hole_count, k in _HOLE_GROUPS.items():
        members = [
            (ri, gi)
            for ri, (_, _, glyphs) in enumerate(records)
            for gi, glyph in enumerate(glyphs)
            if glyph[1] == hole_count
        ]
        if not members:
            continue
        features = np.array(
            [records[ri][2][gi][2] for ri, gi in members], dtype=np.float32
        )
        if k == 1:
            labels = np.zeros(len(features), dtype=int)
        else:
            labels = cv2.kmeans(
                features, k, None, criteria, 20, cv2.KMEANS_PP_CENTERS
            )[1].ravel()
        for (ri, gi), label in zip(members, labels):
            assignment[(ri, gi)] = f"{hole_count}_{label}"
    return assignment


def _plateaus(sequence, min_frames: int):
    """Runs of one unchanging state lasting at least ``min_frames``, merged if adjacent."""
    sequence = sorted(sequence)
    runs, start = [], 0
    for i in range(1, len(sequence) + 1):
        if i == len(sequence) or sequence[i][1] != sequence[start][1]:
            if i - start >= min_frames:
                runs.append((sequence[start][1], sequence[start][0], sequence[i - 1][0]))
            start = i
    merged = []
    for run in runs:
        if merged and merged[-1][0] == run[0]:
            merged[-1] = (run[0], merged[-1][1], run[2])
        else:
            merged.append(run)
    return merged


def _derive_digit_mapping(per_side: dict, final_score: dict, min_frames: int) -> dict:
    """Map glyph clusters to digits using a side whose plateau count is exactly right.

    The k-th plateau of a monotonically increasing score *is* score k, which labels every
    glyph in it without any reference font.
    """
    for side in ("home", "away"):
        plateaus = _plateaus(per_side[side], min_frames)
        if len(plateaus) != final_score[side] + 1:
            continue
        mapping = {}
        for value, (clusters, _, _) in enumerate(plateaus):
            digits = [int(c) for c in str(value)]
            if len(digits) != len(clusters):
                continue
            for cluster, digit in zip(clusters, digits):
                mapping[cluster] = digit
        if len(mapping) == 10:
            return mapping
    raise RuntimeError(
        "could not self-calibrate: no side produced the expected number of plateaus"
    )


def _longest_valid_chain(candidates: list) -> list:
    """Longest subsequence where each step adds exactly one goal to exactly one side."""
    n = len(candidates)
    best, previous = [1] * n, [-1] * n
    for j in range(n):
        for i in range(j):
            (away_i, home_i) = candidates[i][0]
            (away_j, home_j) = candidates[j][0]
            steps = (away_j - away_i) + (home_j - home_i)
            if away_j >= away_i and home_j >= home_i and steps == 1:
                if best[i] + 1 > best[j]:
                    best[j], previous[j] = best[i] + 1, i
    index = int(np.argmax(best))
    chain = []
    while index != -1:
        chain.append(candidates[index])
        index = previous[index]
    return chain[::-1]


def read_timeline(
    strip: np.ndarray,
    live_mask: np.ndarray,
    final_score: dict,
    sample_fps: float = 2.0,
    min_plateau_s: float = 30.0,
    min_state_s: float = 3.0,
) -> list[ScoreState]:
    """Recover the scoring timeline from the score-bug strip.

    ``final_score`` is ``{"away": n, "home": n}`` from the API and is used only to
    self-calibrate the digit mapping.
    """
    frames = np.flatnonzero(live_mask)
    records, per_side = [], {"away": [], "home": []}
    for frame in frames:
        for side, columns in SCORE_COLUMNS.items():
            glyphs = _segment_digits(strip[frame], columns)
            if glyphs:
                records.append((side, int(frame), glyphs))

    assignment = _cluster_glyphs(records)
    for ri, (side, frame, glyphs) in enumerate(records):
        clusters = tuple(
            assignment[(ri, gi)] for gi in range(len(glyphs)) if (ri, gi) in assignment
        )
        if len(clusters) == len(glyphs):
            per_side[side].append((frame, clusters))

    mapping = _derive_digit_mapping(
        per_side, final_score, int(min_plateau_s * sample_fps)
    )

    readings: dict[int, dict] = {}
    for side in ("away", "home"):
        for frame, clusters in per_side[side]:
            try:
                value = int("".join(str(mapping[c]) for c in clusters))
            except KeyError:
                continue
            if 0 <= value <= final_score[side]:          # reject impossible readings
                readings.setdefault(frame, {})[side] = value

    paired = [
        (frame, r["away"], r["home"])
        for frame, r in sorted(readings.items())
        if "away" in r and "home" in r
    ]
    candidates = [
        ((state[0][0], state[0][1]), state[1] / sample_fps, state[2] / sample_fps)
        for state in _plateaus(
            [(f, (a, h)) for f, a, h in paired], int(min_state_s * sample_fps)
        )
    ]
    return [
        ScoreState(away=score[0], home=score[1], start_s=start, end_s=end)
        for score, start, end in _longest_valid_chain(candidates)
    ]


def alignment_anchors(timeline: list[ScoreState]) -> list[tuple[float, str]]:
    """Each score change as ``(video_seconds, scoring_side)``.

    These are the points to align against the API's goal events.
    """
    return [
        (later.start_s, "away" if later.away > earlier.away else "home")
        for earlier, later in zip(timeline, timeline[1:])
    ]
