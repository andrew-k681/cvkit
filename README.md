# cvkit

Command-line toolkit for building object-detection datasets out of video.

Getting from footage to a trained detector is mostly not modelling — it is
sourcing frames, deciding which of the thousands deserve a human's attention,
reviewing them without losing track of what you already judged, and keeping the
dataset honest. `cvkit` is the tools for that part.

**The rule the whole toolkit is built on: detectors propose, humans dispose.**
Nothing here writes a label you did not look at.

```
source                 mine                      curate              ship
──────                 ────                      ──────              ────
download  →  frames  →  mine  →  sheets  →  collect  →  verify  →  upload-images
                                     ↑         ↓          ↓          mark-null
                              review-video   record    fix-export  →  train
                                                                      upload-weights
```

## Install

```bash
pip install "cvkit[all] @ git+https://github.com/andrew-k681/cvkit"
```

cvkit has **no runtime dependencies** — you install only what the commands you
use require, and a command whose extra is missing names the one to install.

| extra | brings | needed by | weight |
|---|---|---|---|
| *(base)* | — | `frames` | 0 packages |
| `download` | `yt-dlp` | `download` | 1 package |
| `roboflow-api` | `requests` | `tag` | 5 packages |
| `images` | `numpy`, `opencv-python` | `sheets`, `collect`, `verify`, `record`, `fix-export` | 2 packages |
| `roboflow` | `roboflow`, `requests` | `upload-images`, `mark-null`, `upload-weights` | 33 packages |
| `detect` | `ultralytics`, `torch`, `numpy`, `opencv-python`, `pyyaml` | `mine`, `train`, `review-video` | the big one |
| `all` | everything above | | |

Combine them: `pip install "cvkit[images,roboflow]"`. `tag` has its own light
extra because it talks to the REST API directly, where the SDK would pull in
Matplotlib, Pillow, Typer and a second OpenCV. `download` and `frames` also
need `ffmpeg` on your PATH.

> The `detect` extra installs Ultralytics, which is **AGPL-3.0**, as are its
> pretrained checkpoints and any model you fine-tune from them. cvkit is MIT
> and bundles neither — read [NOTICE.md](NOTICE.md) before you ship.

## Configuration

Paths default to `./data` and `./runs` under the current directory, so one
install serves every project — `cd` in and run. Override with `--root` or
`CVKIT_ROOT`.

```
data/
  raw/          downloaded videos
  frames/       extracted frames, one directory per video
  mining/       scores, contact sheets, verdict files, track caches
    weights/    downloaded checkpoints (keep out of version control)
  backgrounds/  collected negatives
  to_annotate/  collected positives
runs/           training output
```

Credentials come from the environment or a `.env` file — copy `.env.example`:

```dotenv
ROBOFLOW_API_KEY=...
ROBOFLOW_WORKSPACE=your-workspace
ROBOFLOW_PROJECT=your-project
```

That file is read one variable at a time and never sourced, and a key is never
accepted as a flag — it would land in your shell history.

## The loop

### 1. Source frames

```bash
cvkit download                    # urls.txt -> data/raw (video only, no audio)
cvkit frames                      # data/raw -> data/frames, 1 fps
cvkit frames --fps 2 --force
```

Frames are named `<video_id>_mp4-<index>.jpg`, matching what Roboflow produces
when it splits a video itself, so a locally extracted frame and the same frame
inside an export are recognisably the same image — which is what lets mining
skip frames you have already labelled. Change it with `--frame-sep`.

If a platform already holds frames from these videos, its numbering probably
does not start where a fresh extraction does. Recover the offset once:

```bash
cvkit frames --offset clip01=15 --offset clip02=26
cvkit frames --offsets offsets.json          # {"clip01": 15, ...}
```

### 2. Let a detector shortlist

```bash
cvkit mine --export data/my-export --classes person
cvkit mine --conf 0.03 --imgsz 1280          # even more recall
```

Deliberately **low** confidence and **high** resolution: the goal is recall, so
that a frame reported empty really is. Frames already in `--export`, and their
±2 neighbours, are skipped — a neighbour usually holds the same target.

Run it twice with different models and both `scores*.csv` are merged, a frame
counting as empty only if every pass agrees:

```bash
cvkit mine --model yolo11x.pt --rescore scores.csv --out scores_pass2.csv
```

### 3. Review by eye

```bash
cvkit sheets                                  # 6x6 contact sheets of candidates
cvkit collect --reject "bg_002:5,17 bg_004:3" # drop the tiles that were not empty
cvkit verify                                  # re-render survivors at 960x540
cvkit record --bad "3 11 12"                  # record verdicts
```

Two passes on purpose: 36 tiles at 320×180 is fine for triage and demonstrably
too small for the final call, so anything heading into training is looked at
again at a size where you can see it. `record` writes `rejected_frames.txt` and
`verified_ok.txt`, so a frame you have judged never comes back on a sheet —
sheets are cheap to rebuild, your attention is not.

`sheets --mode positive --min-conf 0.35` flips it around and shortlists frames
that *do* contain targets, for annotation.

### 4. Or review a video directly

```bash
cvkit review-video data/raw/clip.mp4 --model runs/baseline/weights/best.pt
```

The active-learning loop without the upload round-trip: step through with
detections drawn on, watch for where the model is unsure or wrong, hit SPACE,
and the clean full-resolution frame lands ready to upload. One sequential
tracking pass runs at startup and is cached, so navigation and "jump to the
next detection" are lookups with no inference; changing the threshold
re-filters that cache instead of re-running it.

```
SPACE save frame       u  undo last save        q / ESC  quit
d / a  ± stride        . / ,  ± one frame       ] / [  ± 10× stride
n / b  next / previous frame with a detection (wraps)
= / -  confidence      o  overlay               h  help
```

A pose checkpoint works here too, and draws its keypoints:

```bash
cvkit review-video data/raw/clip.mp4 --model yolo11s-pose.pt --imgsz 1280
```

Keypoints below `--kpt-conf` (default 0.5) are hidden, because a pose model
emits a whole skeleton whether or not it can see one. Where a table hides the
torso it invents the shoulders at 0.3 while the wrists are real at 0.9, and
drawing both says the opposite of what the model actually found.

`--events` takes a JSON file of frame ranges something else decided are worth
looking at, draws a band while you are inside one, and repoints `n`/`b` at
events instead of detections:

```bash
cvkit review-video data/raw/clip.mp4 --events events.json
```
```json
[{"start": 287, "end": 349, "label": "hand_up"}]
```

cvkit does not compute them. The rule that fires an event is domain logic, and
usually a threshold calibrated to one camera; this only shows you what
something else proposed, so you can judge it. Repointing `n`/`b` is the point:
on busy footage "next frame with a detection" is just "next frame" -- 1786 of
2090 frames on one measured clip.

`--detections` goes further and reviews a pass another tool made, loading no
model at all:

```bash
cvkit review-video clip.mp4 --detections mediapipe.json --events events.json
```
```json
{"names": {"0": "person"},
 "frames": [[[12, 34, 56, 78, 0.91, 0, 7, [[40, 50, 0.86]]]], []]}
```

One entry per frame, in order; `track_id` may be null and the keypoint list may
be empty. The bare list the tracking pass caches is accepted too, so a cached
pass can be handed to someone else as-is. Coordinates are clamped into the
frame and a frame-count mismatch is warned about — overlaying one decode's
boxes on another's frames is the failure this invites.

This exists for licensing. The `detect` extra is Ultralytics and therefore
**AGPL-3.0**; MediaPipe, RTMPose, ViTPose and RF-DETR are Apache-2.0, and a
project avoiding AGPL cannot import Ultralytics even to look at its own
results. With `--detections`, `review-video` runs in an install that has
neither Ultralytics nor torch.

### 4b. Know what it costs

```bash
cvkit bench data/raw/clip.mp4 --model yolo26s-pose.pt --imgsz 1280
```

Published numbers are COCO at 640 on a datacentre GPU; what decides whether a
model fits an edge box is your frames, at your imgsz, on your hardware. Run it
once per model and compare rows — one process per model on purpose, since a
second model in the same process makes the memory figure meaningless.

Latency is a median and p90, never a mean: the tail is what breaks frame rate.
The warmup frames are discarded because the first inferences pay for lazy init
and kernel compilation.

### 5. Keep the dataset honest

```bash
cvkit fix-export data/my-export --dry-run
cvkit fix-export data/my-export
```

Re-run after every fresh export — the fixes live here, not in the downloaded
files, so regenerating a version never loses them. It repairs two things that
silently wreck a run:

- **Mixed polygon/bbox label files.** If any row has more than 5 fields,
  Ultralytics parses the *whole file* as segmentation and reinterprets the
  plain `cls cx cy w h` rows as 2-point polygons — garbage boxes, no error.
- **Near-duplicate images straddling splits.** An eval frame near-identical to
  a train frame inflates your metric; the eval copy is moved into train.

### 6. Ship it

```bash
cvkit upload-images data/backgrounds --batch round2 --tags background-verified
cvkit mark-null --tag background-verified
cvkit tag --tag reannotate --only-boxes
cvkit train --data data/my-export/data.yaml
cvkit upload-weights --version 3 --run runs/baseline
```

`mark-null` exists for a genuinely surprising platform behaviour: an uploaded
image with no annotation stays "unannotated" and **never reaches a dataset
version**, however often you click Add to Dataset. It needs an explicit *empty*
annotation, and Roboflow rejects an empty YOLO `.txt`, so this writes Pascal
VOC with zero `<object>` elements. Background images are not filler — a
detector that has never seen an empty scene will confidently find your class in
one.

`upload-weights` closes the loop: the model you just trained becomes the thing
that pre-labels your next batch.

## Two things cvkit will not do

**Drop checkpoints in your repo.** Ultralytics downloads a bare weight name
into the *current working directory*. Everything that loads a model `chdir`s
into `--weights-dir` (`data/mining/weights`, gitignored) first, resolving a
relative `--model` to absolute so the `chdir` cannot break it.

That covers only the downloads that read the cwd. Ultralytics resolves others
against `utils.WEIGHTS_DIR`, which ships *relative* (`"weights"`) and is used
long after the `chdir` is restored — CLIP on a world model's `set_classes()`
(353 MB, measured landing in a project root) and the AMP probe on `train()`.
cvkit rebinds it to an absolute path at load time.

**Run what it reads.** A `urls.txt` line cannot become a yt-dlp option, a
`data.yaml` carrying a `download:` key is refused rather than handed to
Ultralytics to `exec()`, and a checkpoint is unpickled under Ultralytics'
restricted loader — which cvkit checks is genuinely in force rather than
assuming it.

## Notes from use

- **Fix the seed.** Evaluation splits on a hand-built dataset are small — a
  hundred images is normal — so run-to-run noise is comparable to a real
  improvement. Compare at the same seed or not at all.
- **Keep some branding negatives.** Without title cards in training, a model
  detects a stylised logo as a real object — 0.82 confidence in one measured
  case. `fix-export --drop-graphics` is **off by default** for this reason; use
  it only to clear *duplicate* branding, and keep a few.
- **Roboflow's search pagination is not stable.** Ordering shifts between
  calls, so one sweep both duplicates and misses rows — an observed pass over
  800 images returned 735 rows holding 657 unique ids. cvkit sweeps until the
  set stops growing; if a count looks short, run it again.
- **Geometry is the only marker of which tool drew an object.** There is no
  author field, and `lastSet` records the last edit, not how the object was
  drawn. `tag --only-boxes` keys off the presence of a `points` array.

## Software bill of materials

[`bom.json`](bom.json) is a [CycloneDX](https://cyclonedx.org) 1.6 SBOM of the
full dependency closure, ready for [Dependency-Track](https://dependencytrack.org).

```bash
./scripts/make_bom.sh          # regenerate
./scripts/make_bom.sh --check  # fail if it is stale
```

It is built from a real resolved environment, because declared ranges
(`>=1.24`) are not matchable against advisories. The committed file is a
snapshot: CI regenerates and validates a current one on every push and uploads
it as an artifact, but deliberately does not fail on drift — an unrelated
upstream release would otherwise break every PR. Refresh it at release time.

## Contributing

```bash
git clone https://github.com/andrew-k681/cvkit && cd cvkit
pip install -e ".[dev]"
python -m pytest tests/ -q
```

[AGENTS.md](AGENTS.md) is the contributor guide — architecture, the contract
every command implements, the conventions that must not be broken, and the
platform traps already paid for. Read it before changing code, whether you are
a person or an AI. (`CLAUDE.md` and `.github/copilot-instructions.md` point at
it, so every tool gets the same instructions.)

## Status

Extracted from a working project and generalised. The Roboflow commands lean on
undocumented endpoint behaviour and may need adjusting as the API moves. Issues
and PRs welcome.

## Licence

cvkit is MIT — see [LICENSE](LICENSE). It bundles no model weights and no
third-party code. Your weights are licensed separately and it matters: see
[NOTICE.md](NOTICE.md).
