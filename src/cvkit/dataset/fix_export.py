"""Repair known defects in a YOLO-format export, idempotently.

Re-run this after every fresh export -- the fixes live here, not in the
downloaded files, so regenerating the dataset version does not lose them.

1. Mixed polygon/bbox label files.
   If any row in a label file has more than 5 fields, Ultralytics parses the
   *whole file* as segmentation and silently reinterprets the plain
   `cls cx cy w h` rows as 2-point polygons, producing garbage boxes. We
   convert polygon rows to their bounding box so every file is uniform.

2. Near-duplicate images straddling splits.
   A frame in valid/test that is visually near-identical to a train frame
   inflates the metric. The valid/test copy is moved into train, so the
   evaluation splits stay clean.

3. Optionally, branding frames marked as background (--drop-graphics).
   Off by default, and usually the wrong thing to do -- see drop_graphics.

    cvkit fix-export data/my-export --dry-run
    cvkit fix-export data/my-export
"""
import shutil
from pathlib import Path

import numpy as np

from .. import imageops

SPLITS = ("train", "valid", "test")


def polygon_to_bbox(parts):
    """`cls x1 y1 x2 y2 ...` (normalised polygon) -> `cls cx cy w h`."""
    cls = parts[0]
    pts = np.array([float(v) for v in parts[1:]], dtype=np.float64).reshape(-1, 2)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def fix_polygons(export, dry, splits=SPLITS):
    changed = 0
    for split in splits:
        for lp in sorted((export / split / "labels").glob("*.txt")):
            rows = [r.split() for r in lp.read_text().splitlines() if r.strip()]
            if not any(len(r) > 5 for r in rows):
                continue
            out = [polygon_to_bbox(r) if len(r) > 5 else " ".join(r) for r in rows]
            n_poly = sum(1 for r in rows if len(r) > 5)
            print(f"  [{split}] {lp.name}: {n_poly} polygon row(s) -> bbox")
            if not dry:
                lp.write_text("\n".join(out) + "\n")
            changed += 1
    print(f"polygon fix: {changed} file(s)" + (" (dry run)" if dry else ""))
    return changed


def drop_graphics(export, thresh, dry, splits=SPLITS):
    """Remove zero-box images that are video branding rather than footage.

    Off by default, and think before you turn it on. It was written when 25
    near-identical title cards were leaking across splits as nulls. But a
    handful of *deduplicated* branding negatives is valuable: without them a
    model will happily detect a title-card logo as a person at 0.82
    confidence, so stripping them wholesale is exactly wrong. Enable it only
    to clean up an export carrying duplicate branding, and keep a few.

    Only zero-box images are considered -- a graphic that somehow contains a
    labelled object is left alone.
    """
    dropped = 0
    for split in splits:
        for ip in sorted((export / split / "images").glob("*.jpg")):
            lp = export / split / "labels" / (ip.stem + ".txt")
            boxes = len([r for r in lp.read_text().splitlines() if r.strip()]) if lp.exists() else 0
            if boxes:
                continue
            f = imageops.flatness(ip)
            if f > thresh:
                print(f"  [{split}] {ip.name[:44]}  flatness {f:.2f}")
                if not dry:
                    ip.unlink()
                    if lp.exists():
                        lp.unlink()
                dropped += 1
    print(f"branding frames: {dropped} dropped" + (" (dry run)" if dry else ""))
    return dropped


def fix_cross_split_dupes(export, thresh, dry, splits=SPLITS):
    items = []
    for split in splits:
        for p in sorted((export / split / "images").glob("*.jpg")):
            d = imageops.descriptor(p)
            if d is not None:
                items.append((split, p, d))
    if not items:
        return 0
    X = np.stack([d for _, _, d in items])

    moved = 0
    for i in range(len(items)):
        dist = np.abs(X[i + 1:] - X[i]).mean(1)
        for j in np.where(dist < thresh)[0]:
            j = int(j) + i + 1
            a, b = items[i], items[j]
            if a[0] == b[0]:
                continue
            # keep the train copy where it is; pull the eval copy into train
            src = b if b[0] != "train" else a
            if src[0] == "train":          # neither is train: move the test one
                src = b if b[0] == "test" else a
            if src[0] == "train":
                continue
            print(f"  {a[1].name} [{a[0]}] ~ {b[1].name} [{b[0]}]  d={dist[j - i - 1]:.3f}"
                  f"  -> move {src[1].name} to train")
            if not dry:
                for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
                    s = export / src[0] / sub / (src[1].stem + ext)
                    dst = export / "train" / sub / (src[1].stem + ext)
                    if s.exists():
                        shutil.move(str(s), str(dst))
            moved += 1
    print(f"cross-split duplicates: {moved} image(s) moved" + (" (dry run)" if dry else ""))
    return moved


def counts(export, splits=SPLITS):
    return {s: len(list((export / s / "images").glob("*.jpg"))) for s in splits}


def add_args(ap):
    ap.add_argument("export", help="the YOLO export directory (holds train/ valid/ test/)")
    ap.add_argument("--dup-thresh", type=float, default=0.05,
                    help="mean-abs-difference below which two frames are duplicates")
    ap.add_argument("--drop-graphics", action="store_true",
                    help="remove zero-box branding frames (read drop_graphics first)")
    ap.add_argument("--graphics-thresh", type=float, default=0.30,
                    help="uniform-block fraction above which --drop-graphics removes a frame")
    ap.add_argument("--no-dupes", dest="dupes", action="store_false",
                    help="skip the cross-split duplicate pass")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def run(args):
    export = Path(args.export)
    if not export.exists():
        raise SystemExit(f"no export at {export}")
    splits = tuple(s for s in SPLITS if (export / s / "images").is_dir())
    if not splits:
        raise SystemExit(f"{export} has no train/valid/test image directories")

    print(f"export: {export}\nsplits: {list(splits)}\nbefore: {counts(export, splits)}\n")
    print("-- mixed polygon/bbox labels --")
    fix_polygons(export, args.dry_run, splits)
    if args.drop_graphics:
        print("\n-- branding frames marked as background --")
        drop_graphics(export, args.graphics_thresh, args.dry_run, splits)
    if args.dupes:
        print("\n-- near-duplicates across splits --")
        fix_cross_split_dupes(export, args.dup_thresh, args.dry_run, splits)

    # Ultralytics caches the scanned labels; a stale cache would hide a re-run.
    for c in export.rglob("*.cache"):
        print(f"removed stale cache {c.relative_to(export)}")
        if not args.dry_run:
            c.unlink()

    print(f"\nafter:  {counts(export, splits)}")
    return 0
