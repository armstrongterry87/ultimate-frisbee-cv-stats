# Ultimate Frisbee Computer Vision Statistics Tracker

Automated statistics extraction from ultimate frisbee game film.

**Status: Phase 1 (audit) complete. No pipeline code written yet.** The only substantive
document in this repo today is [REPO_AUDIT.md](REPO_AUDIT.md), which is the technical and
strategic assessment that scopes the build.

---

## The idea in one paragraph

Ultimate has a data gap with a sharp edge. The UFA employs a sideline statistician per game and
publishes throw-level play-by-play — with field coordinates — through a public API. Everyone else
(college, club, youth, high school) has game film and no data at all, and the incumbent manual
stat-keeping app has been abandoned since 2014. So: **the UFA is the training set, not the customer.**
Their paired video and labels are the supervision signal for a model that gives everyone else the
stats they currently tally by hand or not at all.

## What's here

| Path | |
|---|---|
| [REPO_AUDIT.md](REPO_AUDIT.md) | Full audit: what we're building on, what breaks on ultimate footage, the recommendation, and the locked v0 decisions. **Read this first.** |

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
python -m yt_dlp -f "bv[height<=1080][ext=mp4]" -o "../data/sample_game.mp4" "<YOUTUBE_URL>"
```

Video-only stream is deliberate — we don't need audio, and skipping the merge avoids an ffmpeg
dependency. Check the `-F` output for frame rate first: prefer 30fps over 60fps, which doubles
processing cost for no analytical gain.

## Locked v0 decisions

Rationale for each is in [REPO_AUDIT.md](REPO_AUDIT.md) §6.

| Decision | |
|---|---|
| **Field spec** | UFA — 110 m × 48.8 m, 18.3 m endzones (gridiron footprint). Configurable presets. |
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

## License

TBD. Third-party components are MIT (supervision, sports, yt-dlp) and Apache-2.0 (RF-DETR).
