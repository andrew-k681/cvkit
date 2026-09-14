"""Measure what a model costs on your own footage: latency, throughput, memory.

    cvkit bench data/raw/clip.mp4 --model yolo26s-pose.pt --imgsz 1280
    cvkit bench data/raw/clip.mp4 --model yolo26n.pt --frames 300 --device cpu

One process per model: a second model in the same process makes the memory
figure meaningless. Median and p90, not a mean -- the tail is what breaks
frame rate.
"""
import statistics as st
import sys
import time
from pathlib import Path

import cv2

from . import config, detector, paths


def peak_rss_mb():
    """Peak RSS in MB, or None off Unix.

    ru_maxrss is bytes on macOS/BSD and kilobytes on Linux; confusing them
    reports 1.8 GB as 1.8 MB. Imported here, not at module scope, or cli.py
    would catch the ImportError on Windows and blame a missing extra.
    """
    try:
        import resource
    except ImportError:
        return None
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def device_memory_mb(torch, dev):
    """What the accelerator holds, or None when there is nothing to ask."""
    d = str(dev)
    try:
        if d.startswith("mps"):
            return torch.mps.driver_allocated_memory() / (1024 * 1024)  # peak, as CUDA
        if d.startswith("cuda") or d.isdigit():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return None                     # a probe must not fail the benchmark
    return None


def add_args(ap):
    paths.add_root_arg(ap)
    detector.add_model_args(ap, default_imgsz=640)
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=200, help="frames to time (default: 200)")
    ap.add_argument("--warmup", type=int, default=20,
                    help="frames to run and discard first (default: 20); the first "
                         "inferences pay for lazy init and kernel compilation and "
                         "are several times slower than the steady state")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", default=None)
    return ap


def run(args):
    ws = paths.workspace(args)
    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"no video at {video}")

    model, dev = detector.load(args.model,
                               paths.resolve(args.weights_dir, ws.weights), args.device)
    torch = config.require("torch", "detect")
    keep = detector.resolve_classes(model.names, args.classes)

    cap = cv2.VideoCapture(str(video))
    ok, probe = cap.read()
    if not ok:
        raise SystemExit(f"could not read {video}")
    h, w = probe.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    weights = Path(args.model)
    if not weights.exists():
        weights = paths.resolve(args.weights_dir, ws.weights) / args.model

    print(f"{args.model} on {dev}, imgsz {args.imgsz}, source {w}x{h}")
    print(f"timing {args.frames} frames after {args.warmup} warmup...", flush=True)

    decode, infer, stages = [], [], []
    done = stalled = 0
    while done < args.warmup + args.frames:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:                       # short clip: loop rather than report fewer
            stalled += 1                 # consecutive, or a dead seek spins here
            if stalled > 2:
                raise SystemExit(f"{video} stopped decoding after {done} frames")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        stalled = 0
        r = model.predict(frame, conf=args.conf, imgsz=args.imgsz, classes=keep,
                          device=dev, verbose=False)[0]
        t2 = time.perf_counter()
        if done >= args.warmup:
            decode.append((t1 - t0) * 1000)
            infer.append((t2 - t1) * 1000)
            stages.append(r.speed)
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{args.warmup + args.frames}", flush=True)
    cap.release()
    if not infer:
        raise SystemExit("nothing was timed -- --frames must be at least 1")

    infer.sort()
    p90 = infer[int(len(infer) * 0.9)]
    med = st.median(infer)
    mean_stage = {k: st.mean([s[k] for s in stages if s.get(k) is not None])
                  for k in ("preprocess", "inference", "postprocess")}
    dmem = device_memory_mb(torch, dev)

    print(f"\n  latency     median {med:6.1f} ms   p90 {p90:6.1f} ms   min {infer[0]:6.1f} ms")
    print("  stages      " + "  ".join(f"{k[:5]} {v:.1f}" for k, v in mean_stage.items())
          + "  (ms, mean, as Ultralytics reports)")
    print(f"  throughput  {1000 / med:6.1f} fps at the median")
    print(f"  decode      median {st.median(decode):6.1f} ms  (not inference, but you pay it)")
    rss = peak_rss_mb()
    print("  memory      " + (f"peak RSS {rss:7.0f} MB" if rss is not None
                              else "peak RSS unavailable on this platform")
          + (f"   {str(dev)} {dmem:.0f} MB" if dmem is not None else ""))
    if weights.exists():
        print(f"  weights     {weights.stat().st_size / (1024 * 1024):.1f} MB on disk")
    return 0
