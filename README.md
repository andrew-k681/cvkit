# cvkit

Command-line toolkit for building object-detection datasets out of video.

You have some footage and a class you care about. Getting from there to a
trained detector is mostly not modelling — it is sourcing frames, deciding
which of the thousands are worth a human's attention, reviewing them without
losing track of what you already judged, and keeping the dataset itself honest.
`cvkit` is the set of tools for that part, pulled out of a working project so
it can be reused in the next one.

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

cvkit has **no runtime dependencies**. Nothing in the base install imports a
third-party package, so you install only what the commands you actually use
require:

| extra | brings | needed by | weight |
|---|---|---|---|
| *(base)* | — | `frames` | 0 packages |
| `download` | `yt-dlp` | `download` | 1 package |
| `roboflow-api` | `requests` | `tag` | 5 packages |
| `images` | `numpy`, `opencv-python` | `sheets`, `collect`, `verify`, `record`, `fix-export` | 2 packages |
| `roboflow` | `roboflow`, `requests` | `upload-images`, `mark-null`, `upload-weights` | 33 packages |
| `detect` | `ultralytics`, `torch`, `numpy`, `opencv-python`, `pyyaml` | `mine`, `train`, `review-video` | the big one |
| `all` | everything above | | |

Combine them: `pip install "cvkit[images,roboflow]"`. Run a command whose extra
is missing and it tells you exactly which one to install — no guessing.

`tag` gets its own light extra because it talks to the REST API directly; the
official SDK pulls in Matplotlib, Pillow, Typer and a second copy of OpenCV,
which a tagging loop has no use for.

`download` and `frames` also need `ffmpeg`/`ffprobe` on your PATH
(`brew install ffmpeg`).

> Ultralytics and its pretrained checkpoints are **AGPL-3.0**. cvkit itself is
> MIT and does not bundle them — installing the `detect` extra is your choice,
> and the licence obligations of anything you ship with those weights are
> yours to satisfy. cvkit at least keeps the downloads out of your repo: see
> [Weights](#weights-never-land-in-your-repo).

## Configuration

Paths default to `./data` and `./runs` under the current directory, so one
install serves every project — `cd` into a project and run. Override with
`--root` or `CVKIT_ROOT`.

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

Credentials come from the environment or a `.env` file. Copy `.env.example`:

```dotenv
ROBOFLOW_API_KEY=...
ROBOFLOW_WORKSPACE=your-workspace
ROBOFLOW_PROJECT=your-project
```

cvkit reads one variable at a time out of that file and **never sources it** —
sourcing executes the file, which would export every other secret it holds and
run anything that got appended to it. Never pass a key as a command-line flag
either; it lands in your shell history.

## The loop

### 1. Source frames

```bash
cvkit download                    # urls.txt -> data/raw (video only, no audio)
cvkit frames                      # data/raw -> data/frames, 1 fps
cvkit frames --fps 2 --force
```

Frames are named `<video_id>_mp4-<index>.jpg`, matching what Roboflow produces
when it splits a video itself — so a frame you extracted locally and the same
frame inside an export are recognisably the same image. That is what lets the
mining step skip frames you have already labelled. Change it with
`--frame-sep`.

If a platform already holds frames from these videos its numbering probably
does not start where a fresh extraction does. Recover the offset once and
record it:

```bash
cvkit frames --offset clip01=15 --offset clip02=26
cvkit frames --offsets offsets.json          # {"clip01": 15, ...}
```

### 2. Let a detector shortlist

```bash
cvkit mine --export data/my-export --classes person
cvkit mine --conf 0.03 --imgsz 1280          # even more recall
```

Run deliberately at a **low** confidence and a **high** resolution. The goal is
recall: a frame this reports as empty should really be empty. Frames already in
`--export`, and their ±2 neighbours, are skipped — a neighbour of a labelled
frame usually holds the same target and teaches nothing.

Run it twice with different models and both `scores*.csv` files are merged, a
frame counting as empty only if every pass agrees:

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

Two passes on purpose. 36 tiles at 320×180 is fine for triage and
demonstrably too small for the final call — small, distant and low-contrast
targets survive it. Anything heading into training gets looked at again at a
size where you can actually see it.

`record` writes `rejected_frames.txt` and `verified_ok.txt`, so a frame you
have already judged never comes back on a sheet. Sheets are cheap to rebuild;
your attention is not.

`sheets --mode positive --min-conf 0.35` flips the whole thing around and
shortlists frames that *do* contain targets, for annotation.

### 4. Or review a video directly

```bash
cvkit review-video data/raw/clip.mp4 --model runs/baseline/weights/best.pt
```

The active-learning loop without the upload round-trip: step through the video
with detections drawn on, watch for where the model is unsure or wrong, hit
SPACE, and the clean full-resolution frame lands in a folder ready to upload.

One sequential tracking pass runs at startup and is cached to disk, so
navigation and "jump to the next detection" are pure lookups with no inference.
Adjusting the confidence threshold re-filters the cache instead of re-running.

```
SPACE save frame       u  undo last save        q / ESC  quit
d / a  ± stride        . / ,  ± one frame       ] / [  ± 10× stride
n / b  next / previous frame with a detection (wraps)
= / -  confidence      o  overlay               h  help
```

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
  plain `cls cx cy w h` rows as 2-point polygons — producing garbage boxes with
  no error message. Polygon rows are converted to their bounding box so every
  file is uniform.
- **Near-duplicate images straddling splits.** A valid/test frame that is
  visually near-identical to a train frame inflates your metric. The eval copy
  is moved into train so the evaluation splits stay clean.

### 6. Ship it

```bash
cvkit upload-images data/backgrounds --batch round2 --tags background-verified
cvkit mark-null --tag background-verified
cvkit tag --tag reannotate --only-boxes
cvkit train --data data/my-export/data.yaml
cvkit upload-weights --version 3 --run runs/baseline
```

`mark-null` exists because of a genuinely surprising platform behaviour: an
uploaded image with no annotation stays "unannotated" and **never reaches a
dataset version**, no matter how many times you click Add to Dataset. It needs
an explicit *empty* annotation to record "this frame deliberately contains
nothing". Roboflow rejects an empty YOLO `.txt` as an unrecognised format, so
this writes Pascal VOC with zero `<object>` elements.

Background images are not filler. A detector that has never seen an empty scene
will confidently find your class in one.

`upload-weights` closes the loop: the model you just trained becomes the thing
that pre-labels your next batch of frames.

## Weights never land in your repo

Ultralytics downloads a bare weight name into the **current working
directory**. Run a script from a repo root and you have just added a hundred
megabytes of AGPL-licensed checkpoint to that repo, and probably committed it.

Everything in cvkit that loads a model `chdir`s into `--weights-dir`
(`data/mining/weights` by default, gitignored) for the duration of the load,
and resolves a relative `--model` path to absolute first so the `chdir` cannot
break it.

## Notes from use

- **Fix the seed.** Evaluation splits on a hand-built dataset are small — a
  hundred images is normal — so run-to-run noise is comparable to a real
  improvement. Compare runs at the same seed or not at all.
- **Keep some branding negatives.** Without title cards and end graphics in
  the training set, a model will confidently detect a stylised logo as a real
  object — 0.82 confidence, in one measured case. `fix-export --drop-graphics` removes them and is **off by
  default** for exactly this reason — use it only to clear out *duplicate*
  branding, and keep a few.
- **Roboflow's search pagination is not stable.** Ordering shifts between
  calls, so one sweep returns duplicates and misses rows — an observed pass
  over 800 images came back with 735 rows holding 657 unique ids. cvkit sweeps
  repeatedly until the set stops growing. It is a workaround, not a guarantee;
  if a count looks short, run it again.
- **Geometry is the only marker of which tool drew an object.** There is no
  author field, and `lastSet` records the most recent edit, which says nothing
  about how the object was drawn. `tag --only-boxes` uses the presence of a
  `points` array to tell an assisted polygon from a hand-drawn box.

## Software bill of materials

[`bom.json`](bom.json) is a [CycloneDX](https://cyclonedx.org) 1.6 SBOM of the
full dependency closure — every component with a concrete version and
PackageURL, ready to ingest into [Dependency-Track](https://dependencytrack.org)
or any other CycloneDX consumer.

```bash
./scripts/make_bom.sh          # regenerate
./scripts/make_bom.sh --check  # fail if it is stale
```

It is built from a real, fully resolved environment with every extra
installed, because declared ranges (`>=1.24`) are not matchable against
advisories — Dependency-Track needs pinned versions. The environment is
created without `pip`/`setuptools` preinstalled so venv bootstrap does not
appear as a false component. (`setuptools` still appears: `torch` genuinely
requires it.)

The committed file is a **snapshot** of one resolution. CI regenerates and
validates a current one on every push and uploads it as a build artifact, but
deliberately does not fail on drift from the committed copy — an unrelated
upstream release would break every PR. Refresh it deliberately, at release
time. To push each release into Dependency-Track:

```bash
curl -X POST "$DTRACK_URL/api/v1/bom" \
  -H "X-Api-Key: $DTRACK_API_KEY" \
  -F "project=$DTRACK_PROJECT_UUID" \
  -F "bom=@bom.json"
```

## Contributing

```bash
git clone https://github.com/andrew-k681/cvkit && cd cvkit
pip install -e ".[dev]"
python -m pytest tests/ -q
```

[AGENTS.md](AGENTS.md) is the contributor guide — architecture, the contract
every command implements, the conventions that must not be broken, and the
platform traps already paid for. Read it before changing code, whether you are
a person or an AI assistant. (`CLAUDE.md` and
`.github/copilot-instructions.md` just point at it, so every tool gets the same
instructions.)

## Status

Extracted from a working project and generalised. The Roboflow commands lean
on undocumented endpoint behaviour and may need adjusting as the API moves.
Issues and PRs welcome.

## Licence

cvkit is MIT — see [LICENSE](LICENSE). It bundles no model weights and no
third-party code.

**Your model weights are licensed separately, and it matters.** The `detect`
extra installs Ultralytics, which is AGPL-3.0, as are its pretrained
checkpoints — and a model you fine-tune from them inherits that. See
[NOTICE.md](NOTICE.md) before shipping anything built with this.
