# REPO_AUDIT.md

Audit of `./supervision` as a foundation for an ultimate frisbee analytics product.

**Date:** 2026-08-30
**Auditor:** Claude (Phase 1 — no code written)
**Repo audited:** `roboflow/supervision` @ `0753191a`, version `0.31.0.dev0` (`origin` = https://github.com/roboflow/supervision.git)

---

## 0. Two things I could not verify, and one correction to the brief

**The brief had unfilled placeholders.** The repo URL and the sample-video path were left as
`[<PASTE REPO URL...>]` and `[<PATH OR LINK...>]`. I audited what is actually on disk
(`./supervision`, confirmed by `git remote -v` as roboflow/supervision).

**There is no footage.** `./data/` does not exist; there is no `sample_game.mp4`. **Every accuracy
claim in this document is an a-priori estimate from first principles, labelled as such. I have not
run a single frame of ultimate footage through anything.** No number here is measured. Per your
ground rules, I'm not going to invent any.

**Correction to the premise of the audit questions.** Phase 1 asks what the repo "assumes about the
sport (field markings, player counts, jersey formats, camera angles)." It assumes *nothing* about any
sport. `supervision` is not a sports analytics pipeline — it is a general-purpose computer vision
**utility library** with no models, no training code, no weights, and no sport-specific logic
anywhere in its 43,440 lines. I suspect the brief was written expecting `roboflow/sports` (a
different repo, which *does* contain soccer pitch configs, homography, and team classification).
That materially changes the answer to Q5.

---

## 1. What the repo actually does today

`supervision` is Roboflow's open-source CV toolkit: the glue between a model you supply and the
things you want to do around it. MIT licensed, pure-Python, ~43k LOC in `src/`, ~46k LOC of tests.

**There are no entry points.** No CLI, no `main`, no server, no pipeline. It is a library you
`import`. The only executable code is five standalone demo scripts in `examples/`
(`speed_estimation`, `time_in_zone`, `count_people_in_zone`, `heatmap_and_track`, `compact_mask`),
each bringing its own model dependency (`ultralytics` / `rfdetr` / `inference`).

**There are no models and no weights.** `find` for `*.pt`, `*.onnx`, `*.pth`, or anything named
`*train*` returns nothing. `torch` is not a dependency — it appears only as a lazy `import torch`
inside seven connector functions that convert someone else's tensors to NumPy.

### The architecture, in one line

```
model output --from_*()--> sv.Detections --filter/track/slice--> sv.Detections --annotate()--> image
```

Everything hangs off one dataclass, `sv.Detections` (`supervision/src/supervision/detection/core.py:105`):

| field | shape | note |
|---|---|---|
| `xyxy` | `(n, 4)` | boxes — the only required field |
| `mask` | `(n, H, W)` bool, or `CompactMask` | segmentation, optional |
| `confidence` | `(n,)` | optional |
| `class_id` | `(n,)` | optional |
| `tracker_id` | `(n,)` | populated by a tracker, optional |
| `data` | `dict[str, np.ndarray]` | **per-detection extras, row-aligned with `xyxy`** |
| `metadata` | `dict` | collection-level (video name, timestamp) |

That `data` dict is the extension point that matters for us — it is where `jersey_number`,
`team_id`, `has_disc`, `field_xy` live without touching library code.

### Component inventory

| Area | What's there | Path |
|---|---|---|
| **Model connectors** | ~20 `from_*()` classmethods: Ultralytics, Inference, Transformers, Detectron2, MMDetection, SAM / SAM3, YOLOv5, YOLO-NAS, PaddleDet, EasyOCR, NCNN, TensorFlow, DeepSparse | `detection/core.py` |
| **VLM connectors** | Florence-2, PaliGemma, Qwen2.5-VL / Qwen3-VL, Gemini 2.0/2.5/3.5, Moondream, DeepSeek-VL2 | `detection/vlm.py` |
| **Annotators** | ~25, all sharing `annotate(scene, detections) -> scene`: Box, Mask, Polygon, Halo, Ellipse, Label, RichLabel (TTF/unicode), Trace, Heatmap, Blur, Pixelate, Icon, Dot, Triangle… | `annotators/core.py` |
| **Tracking** | `ByteTrack` — **deprecated, scheduled for removal in 0.31.0, which is the version on disk** | `tracker/byte_tracker/` |
| **Zones & counting** | `LineZone` (in/out crossings, per-class), `PolygonZone` (anchor-in-polygon occupancy), `DetectionsSmoother` | `detection/line_zone.py`, `detection/tools/` |
| **Tiled inference** | `InferenceSlicer` — SAHI-style overlapping tiles, batched callback, offset-back + NMS merge | `detection/tools/inference_slicer.py:104` |
| **Datasets** | `DetectionDataset` with lazy image loading; `from_*`/`as_*` for COCO, YOLO, Pascal VOC, LabelMe, CreateML — doubles as a format converter | `dataset/` |
| **Metrics** | mAP / mAR / precision / recall / F1, COCO evaluator, confusion matrix (`update()`/`compute()`) | `metrics/` |
| **Video I/O** | `VideoInfo`, `VideoSink`, `get_video_frames_generator` (with prefetch), `process_video`, `FPSMonitor` | `utils/video.py` |
| **Keypoints** | `KeyPoints` + Edge/Vertex annotators (pose models) | `key_points/` |
| **`_cv2` shim** | Private OpenCV-compatible surface. Dispatches to real OpenCV when installed (`_cv2.BACKEND_NAME == "opencv"`), else a NumPy/Pillow/PyAV fallback | `_cv2/` |

### What is conspicuously absent

- **No homography / perspective transform.** The `ViewTransformer` class people associate with
  supervision demos lives in an *example script*
  (`supervision/examples/speed_estimation/ultralytics_example.py:24`), is 12 lines long, and calls
  real `cv2.getPerspectiveTransform`. The `_cv2` shim exports **no** `getPerspectiveTransform`,
  `perspectiveTransform`, `findHomography`, `warpPerspective`, or `solvePnP`. Field projection is
  something we write ourselves.
- **No team classification, no re-ID, no jersey OCR.** (EasyOCR has a *connector*, not an integration.)
- **No event detection or state machine of any kind.**
- **No pitch/field keypoint model.**

---

## 2. Sport assumptions — and what breaks on ultimate

**The library encodes zero sport assumptions.** The word "soccer" appears once, as the filename of a
demo JPEG (`assets/list.py:72`); "basketball" once, as a demo MP4. There is nothing to break.

So the real question isn't "what breaks in this repo" but **"what breaks in the naive
soccer/basketball pipeline everyone builds with this repo."** That list is long, and it's where the
domain work actually is.

### 2.1 The field-dimension problem — needs a decision from you

Your brief says "~110m x 37m field with two 23m endzones" but names **UFA broadcast** as the primary
footage source. Those are inconsistent, and the inconsistency is expensive:

| Spec | Total L × W | Endzone depth | Playing proper |
|---|---|---|---|
| Your brief (legacy USAU, 70×40 yd proper / 25 yd EZ) | 110 m × **37 m** | 23 m | 64 m |
| Current USAU / WFDF | 100 m × **37 m** | 18 m | 64 m |
| **UFA** (American football footprint, 120 × 53⅓ yd) | 110 m × **48.8 m** | 18.3 m | 73 m |

UFA plays on gridiron football fields: 80 yards between goal lines, 20-yard endzones, 53⅓ yards wide.
**A UFA field is ~32% wider than the 37 m you specified.** If we encode 37 m and the footage is UFA,
every lateral measurement is compressed by a third — which lands squarely on the advanced metric you
want. "Average cutter separation" computed on a 37 m-wide model of a 48.8 m-wide field isn't a
slightly-noisy metric, it's a systematically wrong one.

*Design consequence:* field geometry goes in a config file with named presets (`ufa`, `usau_current`,
`usau_legacy`, `wfdf`), never a constant in model code.

### 2.2 "No line markings besides the endzone" — probably false for UFA, and that's good news

This is the assumption I'd push back on hardest. Soccer/basketball homography works because those
sports have rich, dense, known markings (center circle, penalty box, arcs) a keypoint model can
regress. A bare club/college ultimate field really does give you only ~8 usable points: 4 field
corners + 4 goal-line/sideline intersections, plus 2 brick marks that are tiny and usually invisible.
Eight well-spread points is plenty for a homography (you need 4) — the actual failure mode is that
they're frequently **out of frame** on a zoomed broadcast shot, not that they're scarce.

But UFA games are played in gridiron and soccer stadiums, and those fields typically retain **yard
lines and hash marks every 5 yards** under the taped ultimate lines. If your sample has them, that's a
dense, regularly-spaced, known-geometry grid — *better* for homography than a soccer pitch, not worse.
It varies by venue (some UFA venues are soccer pitches with different markings), so this has to be
checked against your actual sample rather than assumed in either direction.

**This is the first thing I want to look at when the footage lands.**

### 2.3 The disc is near the resolution floor — this is the headline risk

A disc is 27 cm across. Rough arithmetic for a sideline broadcast frame:

- Full-field framing, 110 m across 1920 px → ~17 px/m → **disc ≈ 5 px face-on, ~1 px edge-on**
- Zoomed to ~40 m of field → ~48 px/m → **disc ≈ 13 px face-on**

Then add: the disc presents as a *thin sliver* whenever it's flying flat away from or toward the
camera (which is most hucks on a sideline angle); it's motion-blurred (a huck moves 20–25 m/s, so a
1/60 s shutter smears it 6–17 px); and it's occluded by the thrower's hand and body for the entire
stall count, which is most of the time it's on screen.

A soccer ball is spherical (constant silhouette), 22 cm, high-contrast, and has multiple large public
datasets. **A disc has none of those properties and, as far as I know, no public labeled dataset.**
Anyone quoting you soccer-ball detection numbers as a proxy is misleading you.

*Design consequence, and I think this is the key architectural call:* **do not build possession on top
of disc detection.** Build it on player kinematics and use the disc as corroboration when visible. See
§5.3.

### 2.4 Where ultimate is *easier* than soccer — exploit this

Three structural properties make this problem more tractable than the soccer equivalent, and a
soccer-derived design throws all three away:

1. **No substitutions during a point.** The 14 players on the field are a fixed set from pull to goal.
   Re-ID only has to hold for ~30–90 seconds, not 90 minutes — and identity can be resolved *once per
   point* rather than continuously. This is a large simplification and should be the backbone of the
   tracking design.
2. **Possession is a stable, rule-enforced state.** The thrower must establish a pivot (so they stop
   moving), holds for up to 10 seconds under a stall count, and has a marker within 3 m. "Who has the
   disc" is a multi-second dwell state, not soccer's sub-second contact events. It's inferable from
   motion alone: *the player who stops, with a defender planted on them, is the thrower.*
3. **Every point starts with a pull** — a hard, unambiguous segmentation boundary with both teams lined
   up on their goal lines. Free point-boundary detection, free "which line was on" snapshot, free
   periodic re-anchor for homography and identity.

### 2.5 Other ultimate-specific breakages in a soccer-derived pipeline

- **Jersey colors.** Club/college ultimate wears anything (white-on-white matchups, tie-dye, dark-on-dark).
  UFA is more disciplined (real kits, contrasting light/dark), so this is less severe for the beachhead
  than for college footage. Color-histogram team assignment is unreliable in general; embedding +
  cluster-per-point is the right approach, seeded by the pull formation — teams start on opposite goal
  lines, which is free supervision for the clustering on every single point.
- **Jersey numbers.** Readable only when a player faces the camera at sufficient zoom — realistically a
  minority of frames on a wide shot, and back numbers only when facing away. Any design that *requires*
  OCR per frame will fail. OCR should be an evidence source voting into a per-point identity assignment,
  with a human-in-the-loop fallback (coach confirms 7 names once per point in the viewer). A
  half-automated product a coach trusts beats a fully-automated one they don't.
- **No set positions.** Soccer pipelines lean on positional priors (this blob near the goal is probably
  the keeper). Ultimate has handlers and cutters, but they're fluid and swap constantly. Drop positional
  priors.
- **The camera is probably not fixed.** Your brief says "single fixed camera," but broadcast UFA is
  typically an *operated* camera that pans and zooms to follow play, with cuts to a second angle and
  replay inserts. If so: homography must be re-estimated per frame (or per keyframe with interpolation),
  and we need shot-boundary detection, or replays get silently ingested as live play and corrupt every
  count. This is the second thing I want to check on the sample, and it's a significant architectural
  fork.
- **Field of view.** With an operated camera following the disc, all 14 players are frequently *not* in
  frame. "Average cutter separation" needs the defender too — so any metric requiring full-field state
  will have real coverage gaps. Worth measuring before promising the metric.
- **Vertical play.** Layouts, skys, and greatests mean players leave the ground and land horizontally.
  Bottom-center anchoring (`sv.Position.BOTTOM_CENTER`, the default for `PolygonZone` and every
  ground-plane projection) assumes feet on the ground. A layout's bottom-center box edge is the player's
  *shoulder*, projecting them several metres downfield. Ultimate has more airborne frames than soccer,
  and they cluster precisely on the highest-value plays — goals, Ds, callahans, i.e. exactly the frames
  we most need to be right. Needs an explicit aspect-ratio-based airborne flag.

---

## 3. Reusable as-is / needs adaptation / dead weight

### Reusable as-is — genuinely valuable, would cost us weeks to rebuild

- **`sv.Detections` as the internal format.** Row-aligned NumPy, a `data` dict for our custom fields,
  merge/select/slice/NMS, ~20 model connectors for free. Adopting it means any detector we swap to later
  (RF-DETR, YOLO, a custom disc model) drops in without touching downstream code. This alone justifies
  the dependency.
- **`InferenceSlicer`** (`detection/tools/inference_slicer.py:104`). Tiled inference is *not optional*
  for the disc at 1080p (§2.3) — it's the difference between a 5 px object and a 30 px object. Supports
  batched callbacks and threading. Directly load-bearing.
- **Annotators.** The entire debug-visualisation and viewer-overlay layer, free. `TraceAnnotator` for cut
  paths, `HeatMapAnnotator` for field occupancy, `RichLabelAnnotator` for jersey/name labels.
- **`metrics/`.** Real COCO mAP for honestly evaluating our own player and disc detectors — which we need
  on day one to report accuracy rather than vibes.
- **`dataset/`.** COCO/YOLO/VOC round-tripping. We will be labeling a disc dataset and shuttling it
  between labeling tools and training; this is the format converter.
- **Video I/O** (`get_video_frames_generator` with prefetch, `VideoSink`, `VideoInfo`).

### Needs adaptation

- **`LineZone`** — right idea for goal-line crossings, wrong space. It operates in *image* coordinates
  with a fixed line; with a panning camera the goal line moves every frame. Use the concept, apply it in
  projected field coordinates after homography.
- **`PolygonZone`** — same: useful for endzone occupancy, but in field space. Note its anchor-based test
  is explicitly not a true geometric overlap test (the docstring says so at
  `detection/tools/polygon_zone.py:33-40`).
- **`DetectionsSmoother`** — fine for boxes; our trajectory smoothing wants to be in field space with a
  motion model, not image space.
- **`ViewTransformer` from the example** — 12 lines, needs to become robust: RANSAC `findHomography`,
  temporal smoothing, per-frame re-estimation, and a confidence signal so we can *drop* frames where the
  homography is unreliable rather than emit garbage coordinates.

### Dead weight for us

- **`sv.ByteTrack` — deprecated and scheduled for removal in 0.31.0, the exact version cloned here.** It
  still resolves via a lazy `__getattr__` shim (`__init__.py:311`) but the removal notice is explicit
  (`tracker/byte_tracker/core.py:26-40`): use `ByteTrackTracker` from the separate `trackers` package
  (`update_with_detections()` → `update()`). **Do not build our tracking on `sv.ByteTrack`.** Anything
  written against it breaks on the next release.
- `supervision.keypoint` (dead alias for `key_points`), OBB/oriented boxes, GeoTIFF/rasterio support,
  `CompactMask` RLE (unless we go segmentation-heavy), most VLM connectors, `from_easyocr`,
  `ClassificationDataset`, and the `_cv2` fallback (we'll install real OpenCV anyway — we need
  `findHomography`, which the shim does not provide).

---

## 4. Dependencies, hardware, license

**License: MIT** (`LICENSE.md`, Copyright 2022 Roboflow). No copyleft, no attribution burden beyond the
notice, commercial use fine. **Clean for a startup.** Worth noting the *models* we add later carry their
own licenses, and that's where the real trap is: Ultralytics YOLOv8/v11 is **AGPL-3.0**, which for a
hosted commercial product means either a paid Ultralytics license or a non-AGPL detector. RF-DETR
(Apache-2.0) is the safer default and is Roboflow's own. This is a decision to make *before* we train on
YOLO, not after.

**Runtime dependencies** (`pyproject.toml:48-59`) — light, no ML framework:
`av>=14.2`, `defusedxml`, `matplotlib`, `numpy`, `pillow`, `pydeprecate`, `pyyaml`, `requests`, `scipy`,
`tqdm`. Optional extras: `metrics` (pandas), `geotiff` (rasterio). Python 3.10–3.14 (CI tests 3.10–3.13
across Linux, Windows, macOS).

**Notably absent: `opencv-python`.** Replaced by the private `_cv2` shim, which uses real OpenCV when
installed and a NumPy/Pillow fallback otherwise. We'll install real OpenCV regardless — the fallback
lacks the homography functions we need.

**Hardware:** the library itself is **pure CPU** — no GPU code, no CUDA, no realtime assumptions. CI is
CPU-only. All GPU cost in our system comes from the models *we* add. For a 90-minute 1080p game at 30 fps
(162,000 frames), running detection on sampled frames plus tiled slicing for the disc will need a GPU to
be practical; a CPU fallback will be hours, not minutes, and should be documented as such rather than
pretended away. I'll measure this in Phase 2 rather than guess further.

**Maturity:** the repo is healthy — 46k LOC of tests mirroring `src/`, pre-commit with mypy, multi-OS CI,
codecov, active release cadence. A well-maintained dependency, not a research dump.

---

## 5. Recommendation

### 5.1 Verdict: **use as a component. Do not fork. Do not treat it as the foundation.**

There is nothing to fork *from* — no models, no pipeline, no sport logic, no entry point. Forking 43k
lines of well-maintained upstream utility code we don't intend to modify means owning a maintenance
burden and diverging from a dependency that ships useful improvements monthly, in exchange for nothing.

Concretely:

- `pip install supervision` as a normal PyPI dependency (pin it — 0.31 is removing `ByteTrack`).
- Build **our own repo** with the pipeline, the ultimate-specific logic, and the viewer.
- Keep `./supervision` as a **read-only reference checkout**, not the project root. Right now the clone
  *is* the working directory, which will quietly turn into an accidental fork.

### 5.2 On the alternatives you named

**`roboflow/sports`** is the repo the Phase 1 questions seem to describe, and it's the right thing to
**learn from but not depend on**. It has the three pieces supervision lacks — a pitch config with named
keypoints, a `ViewTransformer`, and a `TeamClassifier` (SigLIP embeddings → dimensionality reduction →
KMeans) — and its keypoint-homography pattern is close to what ultimate needs. But it's a demo/showcase
repo, soccer- and basketball-specific, with no stability guarantees. **Copy the patterns, port them to an
ultimate field config, own the code.** A dependency on a demo repo is a liability in a product.

**`YOLOv8 + ByteTrack directly`** is roughly what we'd be doing — but both halves of that phrase have a
catch: Ultralytics YOLOv8 is AGPL (§4), and `sv.ByteTrack` is being removed (§3). The right modern
spelling is **RF-DETR (Apache-2.0) or a licensed detector + the `trackers` package + supervision as the
data-model and utility layer.**

**Replace with a different foundation entirely?** No. There is no off-the-shelf ultimate pipeline. The
differentiated work — disc/possession inference, ultimate's event taxonomy, per-point identity, an
ultimate-specific field model — is work nobody has published, which is precisely why it's a defensible
beachhead. supervision saves us the undifferentiated 30%.

### 5.3 The one architectural call I'd make now: possession-first, disc-second

Given §2.3 (disc near the resolution floor) and §2.4 (possession is a rule-enforced dwell state), I'd
invert the obvious pipeline. Rather than `detect disc → find nearest player → possession`:

1. Track all players; project to field coordinates.
2. Infer the **thrower** from kinematics: the player who decelerates to a pivot with a defender planted
   within ~3 m, held for ≥1 s. Ultimate's rules make this signature unusually clean.
3. Detect **throw events** as the possession-state transition plus a receiver's catch-and-stop, rather
   than by tracking the disc through its flight.
4. Use disc detections, where available, to **confirm and time-align** events and disambiguate contested
   cases — not to carry the pipeline.

This degrades gracefully: at 40% disc recall a disc-first pipeline produces 40% of the events; a
possession-first pipeline produces most events with 40% of them independently confirmed. It also means we
ship something useful before the disc detector is good, and the disc detector improves the product
monotonically rather than gating it.

I'd want to validate this against your sample before committing, and I'd want your read on the pivot
heuristic specifically — you'll both know failure cases I won't (fast-break situations after a turn where
the thrower never really sets, for one).

### 5.4 On the advanced metric: I'd argue for huck completion % by yardage

You asked me to pick one and argue. **Huck completion % by distance bucket:**

- It needs: throw event, catch/turnover outcome, and the field positions of two points (release,
  arrival). Every one of those is already required by the core MVP. **Marginal cost ≈ a `GROUP BY`.**
- It tolerates homography noise. Bucketing into 15–25 m / 25–35 m / 35 m+ means a ±2 m projection error
  rarely changes the answer.
- It's immediately legible and actionable to a coach, and it settles a recurring argument ("we should
  huck less") that nobody currently has data for.

**Average cutter separation**, by contrast, requires: correct defender–cutter *pairing* (an unsolved
assignment problem), both players simultaneously in frame (§2.5 — frequently false on an operated
camera), metric-accurate lateral distances (exactly what §2.1's field-width error destroys), and a
defensible definition of "on the throw" (release frame? decision frame? they differ by ~0.5 s and the
separation changes a lot in that window). Each is a research question; stacked, it's a metric we'd ship
with error bars wider than the signal.

Recommendation: **ship huck completion % in v0; put cutter separation on the roadmap** once homography
accuracy is measured and we know the full-field coverage rate. Happy to be overruled — if you think
separation is the thing that actually sells, say so and I'll scope it honestly rather than quietly
delivering a weak version.

---

## 6. Blocking questions for Phase 2 — answered 2026-08-30

| # | Question | Decision |
|---|---|---|
| 1 | Field spec (§2.1) | **UFA — 110 m × 48.8 m, 18.3 m endzones.** Ships as the default preset; geometry stays configurable in `field.yaml`. |
| 2 | Camera: fixed or operated? (§2.5) | **ANSWERED — directed broadcast.** User confirmed by watching a full UFA game (2026-08-30): hard cuts, replays, and the camera stays tight on the handler rather than holding a full-field shot. Worse than "operated pan/zoom." See §8 for the scope consequences. |
| 3 | Sample video + venue markings (§2.2) | **Still outstanding. This is the live blocker.** |
| 4 | Manual labeling (§2.3) | **Yes — bootstrap set from the sample game.** A few hundred labeled frames, honest accuracy on a held-out split, no extrapolated numbers. |
| 5 | Roster (jersey → name) | Not answered. *Proceeding on: no roster; the viewer lets a coach map track-clusters to names once per point.* |
| 6 | AGPL (§4) | **Not acceptable. Default to RF-DETR (Apache-2.0).** No Ultralytics YOLO in the shipped path. |
| 7 | Dev GPU | Not answered. *Proceeding on: GPU for dev, documented slow CPU fallback.* |

### What Phase 2 is gated on

Everything downstream of frame extraction needs footage. `./data/sample_game.mp4` does not exist and the
path was never filled in. The moment it lands, the first three things I run — before any pipeline work —
are:

1. **Shot-boundary and camera-motion profile.** How many cuts, how much pan/zoom, how much replay
   content. Settles Q2 and the homography architecture.
2. **Venue markings survey.** Sample frames across the game; establish whether gridiron yard lines are
   visible and usable as homography keypoints, or whether we're on corner-only.
3. **Disc pixel-size measurement.** Hand-measure the disc across ~50 frames at varied zoom levels to
   replace §2.3's arithmetic with real numbers. This decides whether the disc detector is viable at all or
   whether v0 is possession-only.

---

## 7. Addendum (2026-08-30): the footage is a directed broadcast

The user confirmed from a full UFA game that the footage is a directed broadcast — hard cuts, replay
inserts, and a camera that stays tight on the handler instead of holding a full-field shot. This is the
hardest camera case, and it splits the MVP cleanly along an axis worth naming:

**A disc-following broadcast camera is bad for spatial analytics and good for event analytics.**

| MVP deliverable | Impact |
|---|---|
| Per-player box score (touches, completions, throwaways, drops, blocks, goals, assists) | **Survives, arguably improves.** The director's job is keeping the disc in frame — free attention-tracking on exactly the events we score. |
| Disc detection | **Substantially easier than §2.3 estimated.** A handler-tight shot spans ~20 m across 1920 px ≈ 96 px/m → disc ≈ 26 px, not 5 px. Ordinary small-object detection, not the resolution floor. Reverts to small/blurred during hucks. |
| Point-by-point breakdown | **Survives, with a large shortcut** — see "free ground truth" below. |
| Player tracking / re-ID | **Degrades hard.** Cuts sever every track; typically only the thrower, the mark, and 2–4 nearby players are in frame. "Track all 14" is off the table. §2.4's per-point identity simplification still helps but must now re-establish after every cut. |
| Field homography | **Much harder.** Per-frame estimation mandatory, and tight handler shots often contain *no* field landmarks at all. Requires chaining optical-flow camera motion between landmark-visible frames, accumulating drift. |
| Cutter separation | **Dead.** Confirms §5.4 — huck completion % is the right v0 metric. |

### Two consequences

**Replay detection is mandatory, not optional.** A replayed goal ingested as live play double-counts.
A box score that reports 4 goals where a player scored 3 is worse than no box score — it's the failure
mode that permanently destroys a coach's trust in the product.

**Broadcast graphics are free ground truth.** OCR on the on-screen score bug yields point boundaries,
running score, and scoring events at near-perfect accuracy with no CV inference. This nearly solves the
point-by-point breakdown outright, and gives a cheap replay signal (graphics typically change or
disappear during replay inserts). Fixed-camera team film does *not* provide this.

### Strategic flag: broadcast may be the wrong primary corpus

The customer is a coach, and **their** games aren't broadcast. UFA broadcasts a few hundred games a year;
a college coach — the larger market — has zero broadcast footage and a drive full of fixed-tripod
sideline film they shot themselves: wide, static, whole field, no cuts, no replays, no score bug. That
footage is far easier for everything in the degraded column above *and* it is the footage the paying
customer actually possesses.

Optimizing for broadcast risks building a product that works beautifully on a corpus the buyer doesn't
have. Recommendation: develop against **both**, with a config flag selecting camera mode — they need
different homography strategies and different tracking assumptions, and one pipeline pretending to
handle both is how this gets messy. Treat fixed team film as the primary target and broadcast as the
harder second case.

### Scope consequence for v0

Shift the MVP's center of gravity from the spatial product to the **event/box-score product** — which is
also the better beachhead on its own merits: it automates the clipboard-and-tally work teams already do
by hand, which is a job a coach already pays for in volunteer hours.

---

## 8. Bottom line

`supervision` is a good dependency and a poor foundation, and those aren't in tension — it was never
meant to be a foundation. Take `Detections`, `InferenceSlicer`, the annotators, and the metrics; skip
`ByteTrack`; write the ultimate-specific 70% ourselves.

The risk ranking has changed since §7. The disc is **no longer** the top risk — broadcast framing puts it
at ~26 px, which is tractable. The top risks are now, in order:

1. **Camera motion and cuts** — severs tracking, and makes homography a per-frame problem with
   landmark-free stretches to bridge.
2. **Replay contamination** — a correctness risk, not an accuracy risk. Must be solved before any box
   score is shown to a coach.
3. **Corpus mismatch** — building for broadcast when the buyer owns fixed-camera film (§7).

The field spec drops to a solved config question.

Correspondingly, v0 should lead with the **event/box-score product** (which the broadcast camera
preserves and the score bug partly labels for free) rather than the spatial product (which it degrades).

**Next step: drop the sample game into `./data/sample_game.mp4`, and get a piece of fixed-camera team
film if one is reachable.**
