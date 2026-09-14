"""Turn `cvkit mine` scores into contact sheets for human review, then collect
the frames you approved.

Nothing here decides what a background image is -- the detector only proposes.
You look at the sheets, note the tile numbers that still contain a target, and
pass them to --reject.

    # sheets of background candidates (the detector found nothing)
    cvkit sheets

    # sheets of frames that DO contain targets, for annotation
    cvkit sheets --mode positive --min-conf 0.35

    # collect what survived (background mode also writes empty .txt labels,
    # which is how YOLO spells "this image deliberately contains nothing")
    cvkit collect --reject "bg_002:5,17 bg_004:3"

    # re-render the collected set large enough to actually judge, then record
    cvkit verify
    cvkit record --bad "3 11 12"

Why two review passes: 36 tiles at 320x180 is fine for triage but demonstrably
too small for the final call -- small, distant and low-contrast targets
survive it. Anything heading into training gets looked at again at 960x540.
"""
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from .. import imageops, naming, paths

TILE = (320, 180)
COLS, ROWS = 6, 6
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------- loading


def score_passes(mining):
    """Every scores*.csv in the mining directory is one detector pass."""
    return sorted(Path(mining).glob("scores*.csv"))


def merge_passes(mining):
    """Union of all passes, taking the most cautious answer for each frame."""
    passes = score_passes(mining)
    if not passes:
        raise SystemExit(f"no scores*.csv in {mining} -- run `cvkit mine` first")
    merged = {}
    for csv_path in passes:
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                r["idx"] = int(r["idx"])
                r["n_det"] = int(r["n_det"])
                r["max_conf"] = float(r["max_conf"])
                cur = merged.setdefault(r["frame"], dict(r, passes=0))
                cur["passes"] += 1
                # a frame counts as empty only if EVERY pass that saw it agrees
                cur["n_det"] = max(cur["n_det"], r["n_det"])
                cur["max_conf"] = max(cur["max_conf"], r["max_conf"])
    print(f"{len(merged)} frames from {len(passes)} pass(es): "
          f"{[p.name for p in passes]}")
    return list(merged.values())


def frame_path(frames_dir, row):
    return Path(frames_dir) / row["video"] / row["frame"]


def select(rows, frames_dir, mode, min_conf, edge_trim=0, max_flat=1.0, sep=naming.DEFAULT_SEP):
    if mode == "background":
        sel = [r for r in rows if r["n_det"] == 0]
    else:
        sel = [r for r in rows if r["max_conf"] >= min_conf]
    print(f"  {len(sel)} after detector filter")

    if edge_trim:
        # Intro and end cards sit at the ends of every video and are all
        # near-identical, so they would otherwise dominate the candidate pool.
        span = {}
        for vd in Path(frames_dir).iterdir():
            if not vd.is_dir():
                continue
            ids = [p[1] for p in
                   (naming.try_split(f.name, sep) for f in vd.glob("*.jpg")) if p]
            if ids:
                span[vd.name] = (min(ids), max(ids))
        before = len(sel)
        sel = [r for r in sel
               if r["video"] in span
               and span[r["video"]][0] + edge_trim <= r["idx"] <= span[r["video"]][1] - edge_trim]
        print(f"  {len(sel)} after trimming {edge_trim} from each end "
              f"({before - len(sel)} dropped)")

    if max_flat < 1.0:
        before = len(sel)
        sel = [r for r in sel
               if imageops.flatness(frame_path(frames_dir, r)) <= max_flat]
        print(f"  {len(sel)} after dropping flat/graphic frames ({before - len(sel)} dropped)")

    return sorted(sel, key=lambda r: (r["video"], r["idx"]))


# ---------------------------------------------------------------- sheets


def _label(im, text, org, scale, colour):
    """Text with a heavy dark outline, so it survives any background."""
    cv2.putText(im, text, org, FONT, scale, (0, 0, 0), int(scale * 5) + 1)
    cv2.putText(im, text, org, FONT, scale, colour, max(1, int(scale * 2)))


def build_sheets(sel, frames_dir, prefix, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob(f"{prefix}_*.jpg"):
        old.unlink()
    per = COLS * ROWS
    manifest = []
    for s in range(0, len(sel), per):
        chunk = sel[s:s + per]
        sheet = np.full((ROWS * TILE[1], COLS * TILE[0], 3), 32, np.uint8)
        for k, r in enumerate(chunk):
            im = cv2.imread(str(frame_path(frames_dir, r)))
            if im is None:
                continue
            im = cv2.resize(im, TILE)
            _label(im, f"{k + 1}", (6, 24), 0.8, (0, 255, 255))
            _label(im, f"{r['video'][:11]}-{r['idx']:04d}", (6, TILE[1] - 8),
                   0.4, (255, 255, 255))
            y, x = divmod(k, COLS)
            sheet[y * TILE[1]:(y + 1) * TILE[1], x * TILE[0]:(x + 1) * TILE[0]] = im
        name = f"{prefix}_{s // per + 1:03d}"
        cv2.imwrite(str(outdir / f"{name}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        manifest.append((name, [r["frame"] for r in chunk]))
    with open(outdir / f"{prefix}_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sheet", "tile", "frame"])
        for name, names in manifest:
            for i, fr in enumerate(names, 1):
                w.writerow([name, i, fr])
    print(f"{len(sel)} frames -> {len(manifest)} sheets in {outdir}")
    return manifest


def verify_sheets(dest, outdir, mining, cols=2, rows=3, size=(960, 540)):
    """Re-render an already-collected set big enough to actually judge."""
    dest, outdir = Path(dest), Path(outdir)
    seen = names_in(verified_file(mining)) | names_in(rejected_file(mining))
    ps = [p for p in sorted((dest / "images").glob("*.jpg")) if p.name not in seen]
    outdir.mkdir(parents=True, exist_ok=True)
    if seen:
        print(f"skipping {len(seen)} frame(s) with a recorded verdict")
    if not ps:
        print("nothing left to verify -- every collected frame has a verdict")
        return
    for old in outdir.glob("verify_*.jpg"):
        old.unlink()
    per = cols * rows
    for s0 in range(0, len(ps), per):
        chunk = ps[s0:s0 + per]
        tiles = []
        for k, p in enumerate(chunk, s0 + 1):
            im = cv2.resize(cv2.imread(str(p)), size)
            _label(im, f"{k}  {p.name}", (8, 30), 0.8, (0, 255, 255))
            tiles.append(im)
        while len(tiles) % cols:
            tiles.append(np.full((size[1], size[0], 3), 32, np.uint8))
        sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(str(outdir / f"verify_{s0 // per + 1:03d}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"{len(ps)} collected frames -> {(len(ps) + per - 1) // per} "
          f"verify sheets in {outdir}")


# ---------------------------------------------------------------- verdicts


def rejected_file(mining):
    return Path(mining) / "rejected_frames.txt"


def verified_file(mining):
    return Path(mining) / "verified_ok.txt"


def names_in(path):
    """One frame name per line; missing file means no verdicts yet."""
    path = Path(path)
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def record_names(path, names):
    """Append to a verdict file, keeping it sorted and unique."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(names_in(path) | set(names))) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- collect


def parse_reject(spec):
    """'bg_002:5,17 bg_004:3' -> {('bg_002', 5), ('bg_002', 17), ('bg_004', 3)}"""
    out = set()
    for part in (spec or "").split():
        if ":" not in part:
            raise SystemExit(f"--reject expects sheet:tiles, got {part!r}")
        sheet, tiles = part.split(":", 1)
        for t in tiles.split(","):
            if t.strip():
                out.add((sheet, int(t)))
    return out


def subsample(frames, n, sep=naming.DEFAULT_SEP):
    """Round-robin across source videos, spreading picks over each timeline.

    Backgrounds should cover many scenes, not cluster in whichever video
    happened to have the most empty terrain.
    """
    by_video = {}
    for f in frames:
        parsed = naming.try_split(f, sep)
        by_video.setdefault(parsed[0] if parsed else "", []).append(f)
    for v in by_video:
        by_video[v].sort(key=lambda f: (naming.try_split(f, sep) or ("", 0))[1])
    out, cursor = [], {v: 0.0 for v in by_video}
    videos = sorted(by_video)
    while len(out) < n and any(cursor[v] < len(by_video[v]) for v in videos):
        for v in videos:
            if len(out) >= n:
                break
            lst = by_video[v]
            # stride through each video so picks are spread across its timeline
            stride = max(1, len(lst) * len(videos) / n)
            i = int(cursor[v])
            if i < len(lst):
                out.append(lst[i])
                cursor[v] += stride
    return sorted(out)


def collect(prefix, outdir, reject, dest, frames_dir, mining,
            empty_labels, max_n=0, sep=naming.DEFAULT_SEP):
    man_path = Path(outdir) / f"{prefix}_manifest.csv"
    if not man_path.exists():
        raise SystemExit(f"no manifest at {man_path} -- run `cvkit sheets` first")
    with open(man_path, newline="") as fh:
        man = list(csv.DictReader(fh))
    bad = parse_reject(reject)
    by_name = names_in(rejected_file(mining))
    keep = [m for m in man
            if (m["sheet"], int(m["tile"])) not in bad and m["frame"] not in by_name]
    if by_name:
        print(f"excluding {len(by_name)} frame(s) listed in {rejected_file(mining).name}")
    if max_n and len(keep) > max_n:
        chosen = set(subsample([m["frame"] for m in keep], max_n, sep))
        keep = [m for m in keep if m["frame"] in chosen]
    dest = Path(dest)
    (dest / "images").mkdir(parents=True, exist_ok=True)
    if empty_labels:
        (dest / "labels").mkdir(parents=True, exist_ok=True)
    for m in keep:
        # The manifest is a plain csv, so treat its fields as untrusted: take
        # the basename only, or a crafted row could write outside `dest`.
        frame = Path(m["frame"]).name
        parsed = naming.try_split(frame, sep)
        vid = Path(parsed[0]).name if parsed else ""
        src = Path(frames_dir) / vid / frame
        if not src.exists():
            print(f"  missing source frame, skipped: {frame}")
            continue
        shutil.copy2(src, dest / "images" / frame)
        if empty_labels:
            (dest / "labels" / (Path(frame).stem + ".txt")).write_text("")
    print(f"rejected {len(bad)}, collected {len(keep)} -> {dest}")


# ---------------------------------------------------------------- commands


def _common(ap):
    paths.add_root_arg(ap)
    naming.add_sep_arg(ap)
    ap.add_argument("--mode", choices=["background", "positive"], default="background",
                    help="background = frames the detector found nothing in; "
                         "positive = frames it did (default: background)")
    ap.add_argument("--frames", default=None, help="frame directory (default: data/frames)")
    ap.add_argument("--dest", default=None,
                    help="collected output (default: data/backgrounds or data/to_annotate)")
    return ap


def _dirs(args):
    ws = paths.workspace(args)
    prefix = "bg" if args.mode == "background" else "positive"
    sheets_dir = ws.mining / f"contact_{prefix}"
    default_dest = ws.data / ("backgrounds" if args.mode == "background" else "to_annotate")
    return ws, prefix, sheets_dir, paths.resolve(args.dest, default_dest)


def add_sheets_args(ap):
    _common(ap)
    ap.add_argument("--min-conf", type=float, default=0.35,
                    help="positive mode only: minimum detector confidence")
    ap.add_argument("--limit", type=int, default=0, help="cap how many frames go on sheets")
    ap.add_argument("--edge-trim", type=int, default=20,
                    help="drop frames within N indices of a video's start/end "
                         "(intro and end cards); 0 disables")
    ap.add_argument("--max-flat", type=float, default=0.30,
                    help="drop frames whose uniform-block fraction exceeds this "
                         "(graphics); 1.0 disables")
    return ap


def run_sheets(args):
    ws, prefix, sheets_dir, _ = _dirs(args)
    frames_dir = paths.resolve(args.frames, ws.frames)
    rows = merge_passes(ws.mining)
    sel = select(rows, frames_dir, args.mode, args.min_conf,
                 args.edge_trim, args.max_flat, args.frame_sep)
    if args.limit:
        sel = sel[:args.limit]
    if not sel:
        raise SystemExit("no frames selected -- loosen --min-conf, --edge-trim or --max-flat")
    build_sheets(sel, frames_dir, prefix, sheets_dir)
    return 0


def add_collect_args(ap):
    _common(ap)
    ap.add_argument("--reject", default="", help='tiles to drop, e.g. "bg_002:5,17 bg_004:3"')
    ap.add_argument("--max", type=int, default=0,
                    help="cap the collected set, spread across videos and timelines")
    return ap


def run_collect(args):
    ws, prefix, sheets_dir, dest = _dirs(args)
    collect(prefix, sheets_dir, args.reject, dest,
            paths.resolve(args.frames, ws.frames), ws.mining,
            empty_labels=(args.mode == "background"), max_n=args.max, sep=args.frame_sep)
    return 0


def add_verify_args(ap):
    return _common(ap)


def run_verify(args):
    ws, prefix, _, dest = _dirs(args)
    verify_sheets(dest, ws.mining / f"verify_{prefix}", ws.mining)
    return 0


def add_record_args(ap):
    _common(ap)
    ap.add_argument("--bad", default="",
                    help="1-based indices from the verify sheets that still "
                         "contain a target; everything else is marked verified")
    return ap


def run_record(args):
    ws, _, _, dest = _dirs(args)
    seen = names_in(verified_file(ws.mining)) | names_in(rejected_file(ws.mining))
    ps = [p for p in sorted((dest / "images").glob("*.jpg")) if p.name not in seen]
    if not ps:
        print("nothing pending -- every collected frame already has a verdict")
        return 0
    bad_i = {int(x) for x in args.bad.replace(",", " ").split()}
    unknown = bad_i - set(range(1, len(ps) + 1))
    if unknown:
        raise SystemExit(f"--bad has indices outside 1..{len(ps)}: {sorted(unknown)}")
    bad = [p.name for i, p in enumerate(ps, 1) if i in bad_i]
    good = [p.name for i, p in enumerate(ps, 1) if i not in bad_i]
    record_names(rejected_file(ws.mining), bad)
    record_names(verified_file(ws.mining), good)
    print(f"recorded {len(bad)} rejected, {len(good)} verified ok (totals: "
          f"{len(names_in(rejected_file(ws.mining)))} / "
          f"{len(names_in(verified_file(ws.mining)))})")
    return 0
