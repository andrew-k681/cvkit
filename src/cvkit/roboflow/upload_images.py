"""Upload images to a Roboflow project with per-split placement, via the SDK.

Uses ROBOFLOW_API_KEY directly rather than an interactive session, so it keeps
working unattended -- no OAuth to expire mid-upload.

Images land in a batch and are NOT added to the Dataset: they stay reviewable
until you annotate them and add them deliberately.

Accepted layouts, in the order they are searched:

    <dir>/train/*.jpg        placed in that split
    <dir>/train/images/*.jpg  ditto, for an export-shaped tree
    <dir>/images/*.jpg       what `cvkit collect` writes -- uses --split
    <dir>/*.jpg              loose files -- uses --split

so the output of `cvkit collect` uploads without rearranging anything.

    cvkit upload-images data/backgrounds --batch backgrounds-round2 \
        --tags background-verified
"""
from pathlib import Path

from .. import config

SPLITS = ("train", "valid", "test")


def add_args(ap):
    config.add_roboflow_args(ap)
    ap.add_argument("directory")
    ap.add_argument("--batch", required=True, help="Roboflow batch name to upload into")
    ap.add_argument("--tags", default="", help="comma separated")
    ap.add_argument("--split", default="train", help="for files not in a split folder")
    ap.add_argument("--ext", default="jpg", help="image extension (default: jpg)")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def run(args):
    d = Path(args.directory)
    if not d.is_dir():
        raise SystemExit(f"no directory at {d}")
    ext = args.ext.lstrip(".")
    jobs, seen = [], set()

    def take(paths, split):
        for p in sorted(paths):
            if p.name not in seen:          # a file is uploaded once, not per layout
                seen.add(p.name)
                jobs.append((p, split))

    for s in SPLITS:
        take((d / s).glob(f"*.{ext}"), s)
        take((d / s / "images").glob(f"*.{ext}"), s)
    take((d / "images").glob(f"*.{ext}"), args.split)
    take(d.glob(f"*.{ext}"), args.split)
    if not jobs:
        where = d.resolve()
        raise SystemExit(
            f"no .{ext} found under {where}. Looked in <split>/, "
            f"<split>/images/, images/ and the directory itself.")

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    counts = {}
    for _, s in jobs:
        counts[s] = counts.get(s, 0) + 1
    ws, pr = config.target(args)
    print(f"{len(jobs)} images -> {ws}/{pr}  batch={args.batch}  tags={tags}")
    print(f"  splits: {counts}")
    if args.dry_run:
        return 0

    key = config.api_key(args)
    project = config.rf_project(args)
    ok = fail = 0
    for i, (p, split) in enumerate(jobs, 1):
        try:
            project.single_upload(image_path=str(p), split=split,
                                  batch_name=args.batch, tag_names=tags,
                                  num_retry_uploads=2)
            ok += 1
        except Exception as e:                      # keep going; report at the end
            fail += 1
            # scrub: the SDK puts the key in its URLs, once per failed image.
            print(f"  FAILED {p.name}: {type(e).__name__} "
                  f"{config.scrub(str(e), key)[:90]}")
        if i % 10 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)}  ok={ok} failed={fail}", flush=True)
    print(f"\nuploaded {ok}, failed {fail}")
    print("images are in the batch only -- not in the Dataset until you add them")
    return 1 if fail and not ok else 0
