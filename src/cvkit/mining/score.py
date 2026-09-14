"""Score every extracted frame with a detector, to find frames worth having.

Two uses for the same pass:
  * frames where the detector finds *nothing* are background (negative)
    candidates -- images that teach the model what an empty scene looks like;
  * frames where it finds something are annotation candidates.

Frames already in your export -- and their +/-N second neighbours -- are
excluded, because a neighbour of a labelled frame usually still contains the
same target and adds no information.

The detector is deliberately run at a very low confidence threshold and a high
input resolution: we want high *recall*, so that a frame reported as empty is
really empty. Its output is a shortlist for human review, never a final label.

    cvkit mine
    cvkit mine --conf 0.03 --imgsz 1280 --classes person
    cvkit mine --rescore scores.csv --out scores_pass2.csv   # second opinion
"""
import csv
import time
from pathlib import Path

from .. import detector, naming, paths

FIELDS = ["frame", "video", "idx", "n_det", "max_conf"]


def labelled_frames(export, sep=naming.DEFAULT_SEP, splits=("train", "valid", "test")):
    """(video, idx) pairs already present in a YOLO-format export."""
    out = set()
    export = Path(export)
    for split in splits:
        for p in (export / split / "images").glob("*.jpg"):
            parsed = naming.try_split(p.name, sep)
            if parsed:
                out.add(parsed)
    return out


def add_args(ap):
    paths.add_root_arg(ap)
    naming.add_sep_arg(ap)
    detector.add_model_args(ap, default_imgsz=1280)
    ap.add_argument("--frames", default=None, help="frame directory (default: data/frames)")
    ap.add_argument("--out", default="scores.csv", help="csv name under data/mining/")
    ap.add_argument("--conf", type=float, default=0.05,
                    help="low on purpose: favour recall (default: 0.05)")
    ap.add_argument("--classes", default=None,
                    help="restrict to these classes, names or ids, comma separated "
                         "(default: every class the model has)")
    ap.add_argument("--export", default=None,
                    help="YOLO export whose labelled frames are excluded from mining")
    ap.add_argument("--window", type=int, default=2,
                    help="also skip +/- this many indices around a labelled frame")
    ap.add_argument("--rescore", default=None,
                    help="second opinion: only score frames a previous csv called empty")
    ap.add_argument("--batch", type=int, default=16)
    return ap


def run(args):
    ws = paths.workspace(args)
    frames_dir = paths.resolve(args.frames, ws.frames)
    mining = ws.mining
    sep = args.frame_sep

    frames = sorted(frames_dir.rglob("*.jpg"))
    if not frames:
        raise SystemExit(f"no frames in {frames_dir} -- run `cvkit frames` first")

    skip = set()
    if args.export:
        lab = labelled_frames(args.export, sep)
        skip = {(v, i + k) for v, i in lab
                for k in range(-args.window, args.window + 1)}
        print(f"{len(frames)} frames | {len(lab)} labelled | {len(skip)} in "
              f"+/-{args.window} window", end=" ")

    cands = []
    for p in frames:
        parsed = naming.try_split(p.name, sep)
        if parsed is None or parsed not in skip:
            cands.append(p)
    print(f"| {len(cands)} to score")

    if args.rescore:
        prev_path = Path(args.rescore)
        if not prev_path.is_absolute():
            prev_path = mining / prev_path
        with open(prev_path, newline="") as fh:
            prev = {r["frame"] for r in csv.DictReader(fh) if int(r["n_det"]) == 0}
        cands = [p for p in cands if p.name in prev]
        print(f"second opinion on the {len(cands)} frames {prev_path.name} called empty")
    if not cands:
        raise SystemExit("nothing to score")

    model, dev = detector.load(args.model,
                               paths.resolve(args.weights_dir, ws.weights), args.device)
    keep = detector.resolve_classes(model.names, args.classes)
    print(f"model {args.model} on {dev} at imgsz={args.imgsz}, "
          f"classes={keep if keep is not None else 'all'}", flush=True)

    mining.mkdir(parents=True, exist_ok=True)
    out_path = mining / args.out
    # Write as we go: a long pass that dies at 90% should not lose everything.
    partial = mining / (args.out + ".partial")
    rows = []
    t0 = time.time()
    with open(partial, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i in range(0, len(cands), args.batch):
            batch = cands[i:i + args.batch]
            res = model.predict([str(p) for p in batch], conf=args.conf,
                                imgsz=args.imgsz, classes=keep, device=dev,
                                verbose=False)
            for p, r in zip(batch, res):
                confs = r.boxes.conf.tolist() if r.boxes is not None else []
                parsed = naming.try_split(p.name, sep) or (p.parent.name, -1)
                rows.append({"frame": p.name, "video": parsed[0], "idx": parsed[1],
                             "n_det": len(confs),
                             "max_conf": round(max(confs), 4) if confs else 0.0})
                w.writerow(rows[-1])
            fh.flush()
            done = min(i + args.batch, len(cands))
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(cands)}  {rate:.1f} fps  "
                  f"eta {(len(cands) - done) / rate / 60:.1f} min", flush=True)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["video"], r["idx"])))
    partial.unlink()

    empty = [r for r in rows if r["n_det"] == 0]
    print(f"\nnothing detected at conf>={args.conf}: {len(empty)}/{len(rows)} "
          f"({100 * len(empty) / max(len(rows), 1):.0f}%)  -> background candidates")
    print(f"wrote {out_path}")
    return 0
