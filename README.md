# Ultimate Frisbee Computer Vision Statistics Tracker

Automated statistics extraction from ultimate frisbee game film.

**Status: Phase 2 — the labeling layer works; no detection model exists yet.**

The pipeline can now take a UFA broadcast and the league's public stats feed and line the
two up well enough that every throw in the game lands on an identifiable video frame. That
is the prerequisite for training anything, and it is what the repo does today. It does
*not* detect players, the disc, possession, or throws from pixels. See
[Roadmap](#roadmap) for what stands between here and that.

[REPO_AUDIT.md](REPO_AUDIT.md) is the technical and strategic assessment that scoped the
build and is still the best orientation.

---

## The idea in one paragraph

Ultimate has a data gap with a sharp edge. The UFA employs a sideline statistician per game
and publishes throw-level play-by-play — with field coordinates — through a public API.
Everyone else (college, club, youth, high school) has game film and no data at all, and the
incumbent manual stat-keeping app has been abandoned since 2014. So: **the UFA is the
training set, not the customer.** Their paired video and labels are the supervision signal
for a model that gives everyone else the stats they currently tally by hand or not at all.

The obstacle is that no public labeled dataset exists for a flying disc, and hand-labeling
is prohibitively slow. The API supplies labels, but stamps them in wall-clock time while the
video has no clock. Tying those two timelines together is the problem this code solves.

## What's here

| Path | |
|---|---|
| [REPO_AUDIT.md](REPO_AUDIT.md) | Full audit: what we're building on, what breaks on ultimate footage, the recommendation, and the locked v0 decisions. **Read this first.** |
| `src/ufa_cv/api.py` | UFA stats API client. Event-type codes verified by reproducing the sample game's 24-21 final from the event stream. |
| `src/ufa_cv/video.py` | Single-pass ffmpeg decode into a memory-mapped grayscale strip. |
| `src/ufa_cv/scorebug.py` | Live-play gate — detects the broadcast score bug to separate live action from replays and breaks. |
| `src/ufa_cv/score.py` | Reads the scoreboard to recover the scoring timeline and the alignment anchors. |

## Quickstart

```bash
pip install -r requirements.txt
```

```python
from pathlib import Path
from ufa_cv import api, scorebug, score, video

GAME = "2026-08-08-NY-BOS"
VIDEO = Path("data/sample_game.mp4")
CACHE = Path("cache/scorebug.raw")        # ~1.2 GB, regenerable

# 1. Ground truth from the league feed.
game = api.fetch_games(game_ids=GAME)[0]
events = api.fetch_game_events(GAME)
goals = api.extract_goals(events)                      # 45 goals

# 2. Decode the video once, keeping only the score-bug strip.
strip = video.extract_gray_strip(VIDEO, scorebug.BUG_STRIP, 2.0, CACHE)

# 3. Which frames are live play?
mask = scorebug.live_play_mask(strip)
segments = scorebug.segments(mask, 2.0)                # live / replay / break

# 4. Recover the scoreboard timeline, and with it the alignment anchors.
timeline = score.read_timeline(
    strip, mask,
    {"away": game["awayScore"], "home": game["homeScore"]},
    sample_fps=2.0,
)
anchors = score.alignment_anchors(timeline)            # 45 (video_seconds, side)
```

Step 2 is the slow one — it decodes the whole game and takes minutes. The cache is reused
on subsequent runs, so only the first call pays that cost.

## What the sample game established

Measured on `2026-08-08-NY-BOS` (UFA East Division Championship), 138.2 min at 1080p59.94.

- **The UFA API is a far richer label source than the audit assumed.** 845 events, of which
  577 are throws carrying both endpoints in field coordinates. Every point also records
  `line` — the seven players on the field — which is §2.4's per-point identity
  simplification handed over as data.
- **Coordinates are yards**, X centred on the field at roughly ±26.7 and Y running 0-120
  from the endline with goal lines at 20 and 100. This confirms the UFA field spec
  empirically, so the pipeline adopts the API's convention natively instead of converting.
- **The live-play gate works.** 83.4 of 138.2 minutes are live play; the gate removes
  pre-game, two quarter breaks, halftime, post-game and 41 replay inserts. That is both the
  replay-double-counting fix and a ~40% cut in frames a detector must touch.
- **Video time maps linearly to API wall-clock.** Quarter breaks alone anchor it only to
  ±30 s, but reading the scoreboard recovers all 45 score changes as dense anchors, which
  brings alignment to **5.2 s rms** (42 of 45 within 10 s). The fitted time scale is
  0.99934 — essentially exactly 1.0, confirming the broadcast VOD is an unedited live feed.
  The scoring side agrees with the API on 43 of 45 goals; the two exceptions are adjacent
  and straddle halftime, consistent with two goals logged out of order rather than a misread.
- **Labels are spatially precise and temporally approximate.** A human sideline statistician
  logs these events, so timings carry ~1-3 s of reaction lag no amount of alignment removes.
  Good for box-score supervision; not sufficient for a frame-exact release detector.
- **Bench players defeat appearance-based filtering.** Substitutes stand just past the
  sideline in identical uniforms at similar scale to active players, alongside officials and
  photographers. No team classifier can separate them — it needs an in-bounds geometric
  test, which couples player filtering to homography.

### Verification

Every number below is checked against the API, which is derived independently of the video.

| Check | Result |
|---|---|
| Score changes recovered | 45 / 45 |
| Timeline endpoints | 0-0 → 24-21, matching the API |
| Monotonicity violations | 0 |
| Scoring side vs API | 43 / 45 |
| Alignment residual | 5.2 s rms, 15.7 s max |
| Fitted time scale | 0.99934 |

### Scope limit on the score reader

`score.py` is a **calibration tool, not a general scoreboard reader**. It is handed the
game's final score from the API and uses it to self-calibrate the digit mapping, so it
aligns footage of games the API already covers. It cannot read the score of a game the API
does not have. That is the right trade for generating training data, and it is why the
reader needs no reference font and no OCR dependency.

Three things were load-bearing, each fixing a specific observed failure:

- **Hole count before clustering.** Clustering raw digit bitmaps confuses 7 with 8;
  topology cannot, since 7 encloses no region and 8 encloses two.
- **Mapping derived from long plateaus.** Ordering states by first appearance fails: the
  graphic's fade-in emits garbage reads *before* the real 0-0 and poisons the ordering.
- **Longest-valid-chain selection.** A stateful accept/reject filter deadlocks permanently
  after one missed step; the chain search degrades gracefully.

Geometry note: the strip is cropped at an **even** height. ffmpeg rounds an odd crop height
down and shifts the origin by a row, which silently misaligns every pixel constant here.

## Roadmap

Nothing below exists yet. Ordered by dependency, not by appeal.

1. **Player detection.** Largely free — pretrained detectors already find people well.
2. **Filtering to in-bounds players.** Requires field homography; see the bench-player
   finding above for why nothing cheaper works.
3. **Tracking across cuts.** The broadcast cuts constantly and every cut severs every track.
4. **Possession and throw inference.** Possession-first per the audit: infer the thrower
   from kinematics rather than from the disc. This is where the aligned API labels pay off —
   577 throws with known thrower, receiver and field positions is the evaluation set.
5. **Disc detection.** Needs a hand-labeled dataset built from this footage, because none
   is public. Last, not first.

## Setup

### 1. Reference checkouts (not committed — 3.9 GB of other people's code)

Two third-party repos are used as read-only reference. They're gitignored; recreate them with:

```bash
mkdir -p reference
git clone https://github.com/roboflow/supervision.git reference/supervision
git clone --depth 1 https://github.com/roboflow/sports.git   reference/sports
```

- **`reference/supervision`** — Roboflow's CV utility library (MIT). A dependency, not a foundation:
  it supplies `sv.Detections` as the common data format, `InferenceSlicer` for tiled small-object
  detection, the annotators, and COCO mAP metrics. It contains no models and no sport logic.
- **`reference/sports`** — Roboflow's soccer/basketball demo repo (MIT, ~1.2k LOC). Read-only pattern
  source for `ViewTransformer` (homography), `SoccerPitchConfiguration` (the parametric field-config
  pattern we port to ultimate), and `TeamClassifier`. **Not a dependency** — it's a demo repo written
  against an older supervision API and it won't run as-is.

### 2. Footage (not committed — copyrighted, multi-GB)

Game video goes in `data/`, gitignored. To fetch a UFA broadcast:

```bash
git clone --depth 1 https://github.com/yt-dlp/yt-dlp.git "yt to mp4"
cd "yt to mp4"
python -m yt_dlp -F "<YOUTUBE_URL>"                    # inspect formats first
python -m yt_dlp -f "<FORMAT_ID>" -o "../data/sample_game.mp4" "<YOUTUBE_URL>"
```

Video-only is deliberate — we don't need audio, and skipping the merge avoids an ffmpeg
dependency at download time. Prefer 30fps where the upload offers it; 60fps doubles decode
cost for no analytical gain. The sample game was only published at 60fps above 480p, so it
was taken at 1080p60 and is frame-sampled during processing instead.

Note that the pixel geometry in `scorebug.py` and `score.py` is specific to 1920x1080 and to
that season's graphics package. Other resolutions or a redesigned score bug need it
re-measured.

## Locked v0 decisions

Rationale for each is in [REPO_AUDIT.md](REPO_AUDIT.md) §6.

| Decision | |
|---|---|
| **Field spec** | UFA — 110 m × 48.8 m, 18.3 m endzones (gridiron footprint). Confirmed empirically by the API's coordinate ranges. |
| **Detector** | RF-DETR (Apache-2.0). AGPL is disqualifying, so no Ultralytics YOLO. NVIDIA's LocateAnything is non-commercial-only and also out. |
| **Tracker** | The `trackers` package — **not** `sv.ByteTrack`, which is deprecated and slated for removal in supervision 0.31.0. |
| **Architecture** | Possession-first, disc-second. Infer the thrower from kinematics; use disc detections to confirm, not to carry the pipeline. |
| **Advanced metric** | Huck completion % by yardage bucket, not cutter separation. |
| **v0 center of gravity** | The event/box-score product, not the spatial product. |

## Known constraints

- **The footage is a directed broadcast** — hard cuts, replay inserts, camera tight on the handler
  rather than holding a full-field shot. Good for event analytics, bad for spatial. See §7.
- **Replay detection is mandatory, not optional.** A replayed goal ingested as live play
  double-counts, and a box score that reports the wrong number of goals is worse than no box score.
- **No public ultimate disc dataset exists.** Bootstrap labeling from the sample game.
- **Lighting shifts within a single game.** The sample runs from daylight into floodlit
  night, which compresses the contrast between light and dark kits that a team classifier
  would lean on.

## License

TBD. Third-party components are MIT (supervision, sports, yt-dlp) and Apache-2.0 (RF-DETR).
