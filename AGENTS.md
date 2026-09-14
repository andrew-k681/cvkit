# AGENTS.md

Working notes for AI coding assistants (and humans) in this repo. Read this
before changing code. `README.md` is the user-facing documentation; this file
is about *how the code is built and why*.

## What this is

`cvkit` is a CLI for building object-detection datasets from video: source
frames, shortlist candidates with a detector, review them by hand, repair the
export, push to Roboflow, train.

**The invariant the whole toolkit rests on: detectors propose, humans dispose.**
No command writes a label a person has not looked at. If you are adding a
feature that auto-accepts detector output as ground truth, you are working
against the design — stop and ask.

## Layout

```
src/cvkit/
  cli.py          the COMMANDS table and dispatch — every command is registered here
  config.py       API key + workspace/project resolution; optional-dependency loader
  paths.py        the Workspace class: default directory layout under a project root
  naming.py       frame filename construction and parsing
  detector.py     Ultralytics model loading, device pick, class resolution
  imageops.py     flatness() and descriptor() — shared by mining and fix_export
  train.py
  video/          download.py, frames.py
  mining/         score.py (mine), sheets.py (sheets/collect/verify/record), review_video.py
  dataset/        fix_export.py
  roboflow/       search.py (pagination workaround), upload_images, mark_null, tag, upload_weights
tests/            pure-logic tests, no torch/network/GUI
```

## The command contract

Each command is a module exposing two functions:

```python
def add_args(ap):    # take an argparse.ArgumentParser, add flags, return it
def run(args):       # do the work, return an int exit code (or None for 0)
```

Register it in `COMMANDS` in `cli.py`:

```python
"my-command": ("cvkit.mining.thing", "add_args", "run", "one-line help"),
```

A module may expose several commands under different `add_args`/`run` names —
`mining/sheets.py` provides four (`add_sheets_args`/`run_sheets`, etc.).

`cli.py` imports the module **only after** the user picks that command. Do not
add module imports to `cli.py` and do not build every subparser up front: that
would make `cvkit frames` require torch.

## Non-negotiable conventions

**1. Optional dependencies are imported lazily, through `config.require`.**

cvkit has **zero** runtime dependencies. Everything third-party is an extra:
`images` (numpy, opencv), `detect` (+ ultralytics, torch), `roboflow` (SDK),
`roboflow-api` (requests only), `download` (yt-dlp). Import them *inside* the
function that needs them:

```python
requests = config.require("requests", "roboflow")   # names the extra to install
```

A bare deferred `import roboflow` gives the user a `ModuleNotFoundError`
traceback instead of an install command — that is what `config.require` exists
to stop.

A *module-level* `import cv2` is fine in a module whose command is gated on an
extra (`sheets.py`, `imageops.py`), because `cli.py` catches the ImportError at
dispatch and names the extra. That is why every entry in `COMMANDS` carries the
extra it needs as its fifth field. If you add a command, set that field, or the
user gets a worse error.

**When you add or change a dependency**, update: the extra in `pyproject.toml`,
the fifth `COMMANDS` field, the README table, `NOTICE.md`, and regenerate
`bom.json` with `./scripts/make_bom.sh`. Declare what you *import* — do not
lean on a transitive dependency (ultralytics happens to pull OpenCV; we import
cv2 directly, so we declare it).

**2. Never source a `.env`.** `config.read_env_var` pulls one variable out with
a regex. Sourcing executes the file: it would export every other secret it
holds and run anything appended to it. There is a test asserting the regex does
not match a longer name (`MY_ROBOFLOW_API_KEY` must not satisfy a read of
`ROBOFLOW_API_KEY`).

**3. Never hardcode a workspace, project, or path.** These started life as
scripts with `WORKSPACE = "..."` at the top; that is exactly what was removed.
Resolution order is always flag → environment → `.env`. Paths come from
`paths.Workspace`, which hangs off the project root (cwd, `--root`, or
`CVKIT_ROOT`).

**4. Never accept a secret as a command-line flag.** It lands in shell history.

**5. Model loading goes through `detector.load`.** Ultralytics downloads a bare
weight name into the **current working directory** — run from a repo root and
you have added a hundred megabytes of AGPL-licensed checkpoint to that repo.
`detector.load` chdirs into the weights dir for the duration of the load, and
resolves a relative `--model` to absolute *first* so the chdir cannot break it.
The chdir is restored in a `finally`. If you write new inference code, call
`detector.load`; do not `from ultralytics import YOLO` yourself.

It is also where `ULTRALYTICS_SAFE_LOAD=1` is set, because a `.pt` is a pickle
and Ultralytics loads one with `weights_only=False` unless told otherwise —
`--model something-someone-sent-you.pt` is otherwise code execution. The flag
is read once into a module constant, so it must be set *before* ultralytics is
imported; that is why the `config.require` call sits below it, not above.

**Setting the flag is not evidence it took**, hence `safe_load_gap()` and the
refusal in `load()`. Both halves fail *quietly*: the flag does not exist before
ultralytics 8.4.67 (bisected, not guessed) and its torch probe needs 2.6 —
which is where the `pyproject.toml` floors come from, so do not lower them
without deleting the claim made here. `ULTRALYTICS_SAFE_LOAD=0` skips the
check. `_SafeLoad` is private, so its absence is deliberately *not* a gap: an
upstream rename must not block a working install.

**6. Treat file contents as untrusted input.** Manifest CSVs, label files,
`urls.txt` and a `data.yaml` are parsed, not trusted — they are the files that
get committed, shared and pasted into.

- `collect` takes `Path(row["frame"]).name` before joining it to a destination,
  so a crafted row cannot write outside the output directory.
- `download.read_urls` refuses a line starting with `-`, and `build_command`
  puts `--` in front of the URLs. Either alone would do; yt-dlp has options
  that run commands (`--exec`).
- `train.check_data_yaml` refuses a `download:` key, which Ultralytics runs via
  subprocess or `exec()` when the `val:` images are missing. Parsed with
  `yaml.safe_load`, not pattern-matched: a regex misses `{download: ...}` flow
  style, and a bypassable check is worse than none.

**7. A secret must never reach an error message.** The Roboflow key travels in
the query string — in this toolkit's own calls *and* in the SDK's, which builds
`...?api_key=<key>` itself — so `requests` puts the whole URL in the message of
every error it raises. Uncaught, that lands in a traceback, a CI log, and
whatever gets pasted into a bug report. `config.scrub` is the one place that
blanks it, and every path that prints or raises such a message uses it:

- the direct REST calls, through `roboflow/search.py::_call`, which raises
  `ApiError`;
- the SDK paths — `config.rf_project`, the per-item handlers in
  `upload_images` and `mark_null`, and the `deploy` call in `upload_weights`,
  which was otherwise an unwrapped traceback.

Three details are load-bearing. The `raise` in `_call` sits *outside* the
`except` block, because inside it the original stays on `__context__` even with
`from None` and anything walking the chain prints the key anyway. `ApiError` is
an `Exception`, not a `SystemExit`, so the per-item handlers in the tagging
loops still catch it and the batch keeps going. And the message leads with the
reason, because those handlers print `str(e)[:90]` and one Roboflow image URL
fills that on its own.

## Frame naming — read before touching `naming.py`

Frames are `<video_id><sep><index>.jpg`, default separator `_mp4-`, matching
what Roboflow produces when it splits a video itself. That is what lets `mine`
recognise a locally extracted frame as one already present in an export.

Two traps, both already fixed, both easy to reintroduce:

- **Negative indices keep full width.** `f"{-16:04d}"` gives `-016` and loses a
  digit to the sign; the real datasets use `-0016`. The sign is placed outside
  the padding. A test covers this.
- **Parsing uses `rpartition`, not `partition`** — a video id may itself contain
  the separator.

`naming.split` also strips Roboflow's `_jpg.rf.<hex>.jpg` export suffix. Use
`try_split` when scanning a directory that may hold unrelated files.

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q        # 49 tests, ~0.2s (7 skip without [images])
```

Tests cover the arithmetic that silently corrupts a dataset when wrong: frame
naming round-trips, offset parsing, reject-spec parsing, subsampling spread,
polygon→bbox conversion, and dotenv reading. They also cover the guards in
conventions 5, 6 and 7 — the urls-file parser and the `--` separator, the
`data.yaml` `download:` refusal, the key scrubbing, and `safe_load_gap` — with
stubs standing in for `requests` and `ultralytics`.

They require **no** torch, network or GUI, but they are not free of *every*
extra: seven import a pure function out of `fix_export.py` or `mining/sheets.py`,
which carry a module-level numpy/cv2 import by convention 1, and five need
PyYAML. All twelve `skipif` rather than fail — as an eager import this broke
collection for everyone, unnoticed because CI had never run. Do not tidy those
imports back to the top. Two of the PyYAML five assert only
`raises(SystemExit)`, which a missing import satisfies: without the skip they
would pass for the wrong reason.

Not covered by tests, and needing manual verification when touched:

- `review-video` — needs a window. Its tracking pass and cache can be exercised
  headlessly by calling `review_video.prescan` directly.
- Anything hitting the Roboflow API. Every such command has `--dry-run`; use it.
- `mine` and `train` — need a real model. Point `--model` at a local checkpoint.

CI (`.github/workflows/ci.yml`) runs on 3.9 and 3.13 in two phases, and the
order matters: it installs `[dev]` and runs the `--help` smoke test of all 14
commands *first*, so that check sees a genuine base install, then adds
`[images]` and runs pytest, so nothing skips in CI. Still no torch and no
Roboflow SDK.

The smoke test does not require `--help` to succeed: on a base install the six
gated commands cannot import their module, so they exit 1 by design, naming the
extra. It passes help *or* a named extra and fails anything else, which is what
checks the fifth `COMMANDS` field. A plain `|| exit 1` cannot work here.

## Known platform behaviour worth not rediscovering

- **Roboflow search pagination is unstable.** Ordering shifts between calls, so
  one sweep returns duplicates *and* misses rows — an observed pass over 800
  images returned 735 rows holding 657 unique ids. `roboflow/search.py` sweeps
  repeatedly until the set stops growing twice. It is a workaround, not a
  guarantee. Do not "simplify" it back to a single pass.
- **An unannotated image never reaches a dataset version.** It needs an
  explicit *empty* annotation, and Roboflow rejects an empty YOLO `.txt` as an
  unrecognised format — hence the Pascal VOC with zero `<object>` elements in
  `mark_null.py`.
- **Object geometry is the only marker of which tool drew a box.** There is no
  author field; `lastSet` records the most recent edit and says nothing about
  how the object was drawn. `tag --only-boxes` keys off the presence of a
  `points` array.
- **Ultralytics parses a label file as segmentation if *any* row has >5
  fields**, silently reinterpreting the plain `cls cx cy w h` rows as 2-point
  polygons and producing garbage boxes with no error. That is what
  `fix-export`'s polygon pass prevents.
- **Keep some branding negatives.** Without title cards in training, a model
  detects a stylised logo as a real object at 0.82 confidence. `fix-export
  --drop-graphics` is off by default for this reason — do not make it default.

## Style

Match what is there. Specifically:

- Module docstrings carry the *usage examples* and the reason the module
  exists. Comments explain **why**, not what — most comments in this repo
  record a trap that was hit. Keep that; do not strip them as "noise", and do
  not pad new code with narration.
- Prefer a clear `argparse` flag over a config file or a new dependency.
- Print progress for anything that loops over many files (`i/total ok/failed`,
  flushed). Long passes write output incrementally so a crash at 90% does not
  lose everything.
- Keep bulk operations non-fatal per item: count failures, report at the end,
  do not abort the batch on one bad file.
- Commit messages: imperative subject, then *why*, wrapping at ~72 columns.
  Do not add `Co-Authored-By` or tool-attribution footers to commits or PRs.

## Before you claim it works

Run the tests, run `pyflakes` over `src/`, and exercise the actual command —
`--help` passing is not evidence that a command works. When changing anything
touching frame names or label maths, verify against real data and compare
byte-for-byte against the previous behaviour; that is how both naming bugs
above were caught.
