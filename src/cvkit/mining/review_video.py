"""Step through a video with a detector overlaid, and bank frames worth annotating.

This is the active-learning loop without the upload round-trip: watch where the
model is unsure or wrong, press SPACE, and the clean full-resolution frame lands
in a folder ready to upload.

    cvkit review-video data/raw/clip.mp4
    cvkit review-video data/raw/clip.mp4 --conf 0.15 --stride 25
    cvkit review-video data/raw/clip.mp4 \
        --model runs/v9/weights/best.pt --classes vehicle

A pose checkpoint works the same way and additionally draws its keypoints:

    cvkit review-video data/raw/clip.mp4 --model yolo11s-pose.pt --imgsz 1280

Keypoints below --kpt-conf are hidden, because a pose model emits a full
skeleton whether or not it can see one -- on footage where a table hides the
torso, the shoulders are invented at 0.3 while the wrists are real at 0.9, and
drawing both says the opposite of the truth.

--events takes a JSON file of frame ranges some other tool decided are worth
looking at -- [{"start": 287, "end": 349, "label": "hand_up"}] -- draws a band
while you are inside one, and repoints n/b at events instead of detections.
cvkit does not compute them: the rule that fires an event is domain logic, and
often a threshold calibrated to one camera, which has no place in a general
tool. This only shows you what something else proposed, so you can judge it.

    cvkit review-video data/raw/clip.mp4 --model yolo26s-pose.pt \
        --imgsz 1280 --events events.json

--detections takes a pass some other tool already made, and skips loading a
model at all -- which is how you review a backend cvkit cannot drive. The
`detect` extra is Ultralytics and therefore AGPL; MediaPipe, RTMPose, ViTPose
and RF-DETR are Apache-2.0, and a project avoiding AGPL cannot import
Ultralytics even to look at its own results. Emit this shape from whatever you
are running:

    {"names": {"0": "person"},
     "frames": [[[x1, y1, x2, y2, conf, cls, track_id, [[x, y, v], ...]]], ...]}

one entry per frame, in order; track_id may be null and the keypoint list may
be empty. That is what prescan caches, so the two are interchangeable.

    cvkit review-video clip.mp4 --detections mediapipe.json --events events.json

n/b changing meaning is the trap here: on busy footage "next frame with a
detection" is just "next frame" (1786 of 2090 on one measured clip), so with
events loaded it is the events you want to step through.

Keys (also printed, and shown in-window with h):
    SPACE  save this frame      u  undo last save
    q or ESC quits; so does closing the window, or Ctrl-C in the terminal
    d / a  forward / back by --stride
    . / ,  forward / back one frame
    ] / [  forward / back 10x stride
    n / b  jump to next / previous frame WITH a detection (wraps around the end)
    = / -  raise / lower the confidence threshold
    o      toggle the overlay        h  help        q or ESC  quit

Saved frames are the untouched source frames, never the annotated view.

Detections come from ONE sequential tracking pass made at startup. A tracker
needs consecutive frames, so it cannot be driven by the random seeking this
tool does -- everything is resolved up front and cached to disk, after which
navigation and n/b are pure lookups with no inference at all. --no-track falls
back to lazy per-frame predict (no ids) if you just want a quick look.
"""
import csv
import hashlib
import json
from pathlib import Path

import cv2

from .. import detector, paths

FONT = cv2.FONT_HERSHEY_SIMPLEX
PALETTE = [(0, 255, 0), (0, 165, 255), (255, 128, 0), (255, 0, 255), (0, 255, 255)]
HELP = [
    "SPACE save frame      u undo last save",
    "d/a  +/- stride       ./, +/- one frame",
    "]/[  +/- 10x stride   n/b next/prev detection (wraps)",
    "=/-  confidence       o overlay   h help   q quit",
]


def load_events(path, total):
    """[(start, end, label)] from a JSON file, sorted.

    Parsed, not trusted (this file is generated or hand-edited, and gets
    committed and shared): indices are coerced to int and clamped into the
    video, so a bad range cannot seek outside it.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for e in data:
        a, b = int(e["start"]), int(e["end"])
        if a > b:
            raise SystemExit(f"event {e!r}: start is after end")
        out.append((max(0, a), min(total - 1, b), str(e.get("label", "event"))))
    return sorted(out)


def load_detections(path):
    """(names, per_frame) from a pass another tool made. See the module docstring.

    Parsed, not trusted: the per-frame rows are indexed positionally
    everywhere downstream, so a row that is too short would fail later and
    somewhere unhelpful.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    names = {int(k): str(v) for k, v in data.get("names", {}).items()} or {0: "object"}
    frames = []
    for n, fr in enumerate(data["frames"]):
        row = []
        for b in fr:
            if len(b) != 8:
                raise SystemExit(
                    f"{path}: frame {n} has a detection with {len(b)} fields, "
                    f"expected 8 (x1 y1 x2 y2 conf cls track_id keypoints)")
            x1, y1, x2, y2, conf, cls, tid, pts = b
            row.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls),
                        tid, [[int(x), int(y), float(v)] for x, y, v in pts or []]))
        frames.append(row)
    return names, frames


def cache_file(cache_dir, video, model_path, imgsz, track_conf):
    """One cache per (video, model, imgsz, floor), invalidated by mtime."""
    # v2: cached detections carry keypoints; a v1 cache has a shorter tuple
    key = f"v2|{video.resolve()}|{Path(model_path).resolve()}|{imgsz}|{track_conf}"
    stamp = f"{video.stat().st_mtime_ns}|{Path(model_path).stat().st_mtime_ns}"
    h = hashlib.sha1((key + "|" + stamp).encode()).hexdigest()[:16]
    return Path(cache_dir) / f"{video.stem}.{h}.json"


def prescan(model, video, dev, args, cache_dir):
    """One sequential tracking pass over the whole video, ALL classes.

    The class filter is applied at display time instead of here, so switching
    --classes never costs another pass -- one cache serves every filter.

    A tracker needs consecutive frames, so it cannot be fed by the random
    seeking this tool does for navigation. We therefore resolve every frame up
    front at a low confidence floor and filter later, which also makes n/b
    instant -- a lazy version runs inference during every scan.
    """
    model_path = Path(args.model)
    if not model_path.exists():
        # a downloadable bare name has no stable local mtime to key a cache on
        cache = None
    else:
        cache = cache_file(cache_dir, video, args.model, args.imgsz, args.track_conf)
        if cache.exists() and not args.rescan:
            per_frame = [[tuple(b) for b in fr] for fr in
                         json.loads(cache.read_text(encoding="utf-8"))]
            print(f"tracking pass: cached ({len(per_frame)} frames) {cache.name}")
            return per_frame

    print(f"tracking pass over {video.name} (once, then cached)...", flush=True)
    per_frame = []
    for i, r in enumerate(model.track(source=str(video), stream=True, persist=True,
                                      conf=args.track_conf, imgsz=args.imgsz,
                                      device=dev, verbose=False)):
        boxes = []
        if r.boxes is not None and len(r.boxes):
            ids = (r.boxes.id.int().tolist() if r.boxes.id is not None
                   else [None] * len(r.boxes))
            kp = getattr(r, "keypoints", None)          # None on a detect model
            kxy = kp.xy.tolist() if kp is not None and kp.conf is not None else None
            kconf = kp.conf.tolist() if kxy is not None else None
            # not `i`: that is the frame counter this loop sits inside
            for j, (b, c, k, t) in enumerate(zip(
                    r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                    r.boxes.cls.int().tolist(), ids)):
                pts = ([[int(x), int(y), round(float(pc), 3)]
                        for (x, y), pc in zip(kxy[j], kconf[j])]
                       if kxy is not None else [])
                boxes.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]),
                              round(float(c), 4), int(k), t, pts))
        per_frame.append(boxes)
        if i and i % 500 == 0:
            print(f"  {i} frames", flush=True)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(per_frame), encoding="utf-8")
        print(f"  done: {len(per_frame)} frames -> {cache.name}")
    else:
        print(f"  done: {len(per_frame)} frames (not cached: --model is a "
              f"downloadable name, pass a path to enable caching)")
    return per_frame


def draw(frame, boxes, idx, total, fps, conf, saved, overlay, show_help,
         name, names, msg, kpt_conf=0.5, event=None):
    view = frame.copy()
    if overlay:
        for x1, y1, x2, y2, c, k, tid, pts in boxes:
            col = PALETTE[k % len(PALETTE)]
            cv2.rectangle(view, (x1, y1), (x2, y2), col, 2)
            for px, py, pc in pts:
                if pc >= kpt_conf:
                    # white ring first: a bare dot in the box colour vanishes
                    # against footage that happens to be that colour
                    cv2.circle(view, (px, py), 5, (255, 255, 255), -1)
                    cv2.circle(view, (px, py), 3, col, -1)
            # class id first, as in the YOLO label files you will be writing
            tag = f"{k}:{names.get(k, k)} {c:.2f}"
            if tid is not None:
                tag += f"  #{tid}"
            (tw, th), _ = cv2.getTextSize(tag, FONT, 0.5, 1)
            cv2.rectangle(view, (x1, y1 - th - 6), (x1 + tw + 6, y1), col, -1)
            cv2.putText(view, tag, (x1 + 3, y1 - 4), FONT, 0.5, (0, 0, 0), 1)

    if event:
        n, total_ev, (a, b, lab) = event
        h_, w_ = view.shape[:2]
        cv2.rectangle(view, (0, 0), (w_ - 1, h_ - 1), (0, 0, 255), 6)
        band = f"{lab}  event {n + 1}/{total_ev}  frames {a}-{b}"
        (tw, th), _ = cv2.getTextSize(band, FONT, 0.9, 2)
        cv2.rectangle(view, (0, 58), (tw + 20, 58 + th + 16), (0, 0, 255), -1)
        cv2.putText(view, band, (10, 58 + th + 4), FONT, 0.9, (255, 255, 255), 2)

    best = max((b[4] for b in boxes), default=0.0)
    per_cls = " ".join(f"{names.get(k, k)}:{sum(1 for b in boxes if b[5] == k)}"
                       for k in sorted({b[5] for b in boxes}))
    bar = (f"{name}  frame {idx}/{total}  {idx / fps:6.1f}s   "
           f"det {len(boxes)} (max {best:.2f}) {per_cls}   conf>={conf:.2f}   "
           f"saved {saved}{'' if overlay else '   [overlay off]'}")
    h, w = view.shape[:2]
    cv2.rectangle(view, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(view, bar, (8, 21), FONT, 0.55, (0, 255, 255), 1)
    if msg:
        # Feedback belongs on screen: the terminal is usually hidden, and a
        # silent window is indistinguishable from a hung one.
        cv2.rectangle(view, (0, 30), (w, 58), (0, 0, 0), -1)
        cv2.putText(view, msg, (8, 50), FONT, 0.6, (0, 165, 255), 1)
    if show_help:
        cv2.rectangle(view, (0, h - 22 * len(HELP) - 12), (620, h), (0, 0, 0), -1)
        for i, line in enumerate(HELP):
            cv2.putText(view, line, (8, h - 22 * (len(HELP) - i) + 4),
                        FONT, 0.48, (255, 255, 255), 1)
    return view


def add_args(ap):
    paths.add_root_arg(ap)
    detector.add_model_args(ap, default_imgsz=640)
    ap.add_argument("video")
    ap.add_argument("--out", default=None,
                    help="where saved frames go (default: data/to_annotate_video)")
    ap.add_argument("--conf", type=float, default=0.25, help="display threshold")
    ap.add_argument("--stride", type=int, default=10, help="frames per d/a step")
    ap.add_argument("--start", type=int, default=0, help="first frame")
    ap.add_argument("--scan-limit", type=int, default=0,
                    help="max frames n/b will scan; 0 = the whole video, which "
                         "is cheap when scanning is a lookup, not inference")
    ap.add_argument("--width", type=int, default=1280, help="window width")
    ap.add_argument("--no-track", dest="track", action="store_false",
                    help="skip the tracking pass: lazy per-frame predict, no ids")
    ap.add_argument("--track-conf", type=float, default=0.05,
                    help="floor for the tracking pass; --conf then filters the "
                         "cached result, so =/- needs no re-run")
    ap.add_argument("--rescan", action="store_true", help="ignore the cached pass")
    ap.add_argument("--detections", default=None,
                    help="review a pass another tool made, loading no model; "
                         "lets an Apache-2.0 backend be reviewed without "
                         "importing Ultralytics")
    ap.add_argument("--events", default=None,
                    help="JSON of [{start, end, label}] frame ranges to flag; "
                         "n/b then step events instead of detections")
    ap.add_argument("--kpt-conf", type=float, default=0.5,
                    help="hide pose keypoints below this confidence (default: 0.5); "
                         "a pose model emits a whole skeleton even where it can "
                         "only see a hand")
    ap.add_argument("--classes", "--class", dest="classes", default=None,
                    help="restrict detection to these classes: names or ids, "
                         "comma separated (default: every class the model has)")
    return ap


def run(args):
    ws = paths.workspace(args)
    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"no video at {video}")
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not total:
        raise SystemExit(f"could not read {video}")

    if args.detections:
        # no model, so no torch and no Ultralytics in the process at all
        model = dev = None
        names, loaded = load_detections(args.detections)
    else:
        model, dev = detector.load(args.model,
                                   paths.resolve(args.weights_dir, ws.weights), args.device)
        names, loaded = model.names, None
    keep = detector.resolve_classes(names, args.classes)
    label = ",".join(names[k] for k in keep) if keep else "all"
    print(f"{video.name}: {total} frames @ {fps:.1f} fps")
    print(f"detections {Path(args.detections).name} ({len(loaded)} frames), no model loaded"
          if loaded is not None else f"model {Path(args.model).name} on {dev}")
    print(f"classes {names} -> filter: {label}")

    events = load_events(args.events, total) if args.events else []
    if events:
        print(f"events: {len(events)} from {args.events} "
              f"({', '.join(sorted({e[2] for e in events}))})")

    per_frame = loaded if loaded is not None else (
        prescan(model, video, dev, args, ws.trackcache) if args.track else None)
    if per_frame is not None:
        tracks = {}
        for fr in per_frame:
            for b in fr:
                k, tid = b[5], b[6]
                if tid is not None and (keep is None or k in keep):
                    tracks.setdefault(k, set()).add(tid)
        print("unique tracks: " + (", ".join(
            f"{names.get(k, k)}={len(v)}" for k, v in sorted(tracks.items())) or "0"))
    print("\n".join("  " + h for h in HELP))
    if events:
        print("  n/b step EVENTS, not detections, because --events was given")

    out = paths.resolve(args.out, ws.data / "to_annotate_video")
    (out / "images").mkdir(parents=True, exist_ok=True)
    manifest = out / "saved.csv"
    if not manifest.exists():
        with open(manifest, "w", newline="") as f:
            csv.writer(f).writerow(["file", "video", "frame", "seconds", "n_det", "max_conf"])

    frames, dets = {}, {}                 # small caches, keyed by frame index

    def get(idx):
        if idx not in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if not ok:
                return None
            if len(frames) > 60:
                frames.clear()
            frames[idx] = fr
        return frames[idx]

    def detect(idx, conf):
        if per_frame is not None:               # the tracking pass already did it
            if not 0 <= idx < len(per_frame):
                return []
            return [b for b in per_frame[idx]
                    if b[4] >= conf and (keep is None or b[5] in keep)]
        key = (idx, round(conf, 2))             # --no-track: lazy, and no ids
        if key not in dets:
            fr = get(idx)
            if fr is None:
                return []
            r = model.predict(fr, conf=conf, imgsz=args.imgsz, classes=keep,
                              device=dev, verbose=False)[0]
            out_ = []
            if r.boxes is not None:
                for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                   r.boxes.cls.int().tolist()):
                    out_.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]),
                                 float(c), int(k), None, []))
            if len(dets) > 400:
                dets.clear()
            dets[key] = out_
        return dets[key]

    def scan_events(idx, step):
        """Start of the next/previous event, wrapping. (index or None, message)."""
        starts = [a for a, _b, _l in events]
        later = [a for a in starts if a > idx] if step > 0 else [a for a in starts if a < idx]
        if later:
            return (min(later) if step > 0 else max(later)), ""
        return (starts[0] if step > 0 else starts[-1]), "wrapped around"

    def scan(idx, step, conf):
        """Next/prev frame carrying a detection. Wraps past the end once, so a
        press at the tail of the video moves instead of silently repeating the
        same failed walk. Returns (index or None, message)."""
        limit = args.scan_limit or total
        probe, seen, wrapped = idx + step, 0, False
        while seen < limit:
            if not 0 <= probe < total:
                if wrapped:
                    break
                probe = 0 if step > 0 else total - 1
                wrapped = True
            if detect(probe, conf):
                return probe, ("wrapped around to the start" if wrapped else "")
            probe += step
            seen += abs(step)
            if per_frame is None and seen % (abs(step) * 25) == 0:
                print(f"  scanning... frame {probe}", flush=True)
        return None, (f"no [{label}] detections within {limit} frames "
                      f"-- try '-' (lower threshold) or a bigger --scan-limit")

    idx, conf, overlay, show_help = max(0, args.start), args.conf, True, True
    saved, msg = [], ""
    win = "review"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.width, int(args.width * 9 / 16))

    frame, dirty = None, True
    try:
        while True:
            if dirty:
                frame = get(idx)
                if frame is None:
                    idx = max(0, min(idx, total - 1))
                    continue
                boxes = detect(idx, conf)
                active = next(((n, len(events), e) for n, e in enumerate(events)
                               if e[0] <= idx <= e[1]), None)
                cv2.imshow(win, draw(frame, boxes, idx, total, fps, conf, len(saved),
                                     overlay, show_help, video.stem, names, msg,
                                     args.kpt_conf, active))
                dirty = False

            # Poll rather than block: waitKey(0) sits in native code, so Ctrl-C
            # is never delivered and the window's close button does nothing.
            k = cv2.waitKey(50) & 0xFF
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break                                   # red X closes it
            if k == 255:
                continue                                # no key this tick

            if k not in (ord("n"), ord("b")):
                msg = ""                                # stale after any other key

            if k in (ord("q"), 27):
                break
            elif k == ord(" "):
                name = f"{video.stem}_f{idx:06d}.jpg"
                dest = out / "images" / name
                if dest.exists():
                    msg = f"already saved {name}"
                    print(f"  already saved {name}")
                else:
                    cv2.imwrite(str(dest), frame)       # clean frame, no overlay
                    with open(manifest, "a", newline="") as f:
                        csv.writer(f).writerow([name, video.stem, idx, round(idx / fps, 2),
                                                len(boxes),
                                                round(max((b[4] for b in boxes), default=0.0), 4)])
                    saved.append(dest)
                    msg = f"saved {name}  ({len(saved)})"
                    print(f"  saved {name}  ({len(saved)} total)")
                dirty = True
            elif k == ord("u") and saved:
                last = saved.pop()
                last.unlink(missing_ok=True)
                msg = f"removed {last.name}"
                print(f"  removed {last.name}")
                dirty = True
            elif k == ord("d"):
                idx = min(total - 1, idx + args.stride); dirty = True
            elif k == ord("a"):
                idx = max(0, idx - args.stride); dirty = True
            elif k == ord("."):
                idx = min(total - 1, idx + 1); dirty = True
            elif k == ord(","):
                idx = max(0, idx - 1); dirty = True
            elif k == ord("]"):
                idx = min(total - 1, idx + args.stride * 10); dirty = True
            elif k == ord("["):
                idx = max(0, idx - args.stride * 10); dirty = True
            elif k in (ord("n"), ord("b")):
                fwd = k == ord("n")
                hit, msg = (scan_events(idx, 1 if fwd else -1) if events
                            else scan(idx, args.stride if fwd else -args.stride, conf))
                if hit is not None:
                    idx = hit
                else:
                    print(f"  {msg}")
                dirty = True
            elif k == ord("="):
                conf = min(0.95, round(conf + 0.05, 2)); dirty = True
            elif k == ord("-"):
                conf = max(0.05, round(conf - 0.05, 2)); dirty = True
            elif k == ord("o"):
                overlay = not overlay; dirty = True
            elif k == ord("h"):
                show_help = not show_help; dirty = True
    except KeyboardInterrupt:
        print("\n  interrupted")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{len(saved)} frames saved to {out / 'images'}")
    print(f"manifest: {manifest}")
    return 0
