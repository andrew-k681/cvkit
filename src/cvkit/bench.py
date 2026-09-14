"""Measure what a model costs on your own footage: latency, throughput, memory.

Published benchmarks are COCO at 640 on a datacentre GPU. What decides whether
a model fits an edge box is your frames, at your imgsz, on your hardware --
and on a 2622x1206 CCTV frame at imgsz 1280 that is a different number
entirely.

    cvkit bench data/raw/clip.mp4 --model yolo26s-pose.pt --imgsz 1280
    cvkit bench data/raw/clip.mp4 --model yolo26n.pt --frames 300 --device cpu

Run it once per model and compare rows; one process per model on purpose,
because a second model loaded into the same process makes the memory figure
meaningless.

Latency is reported as a median and p90, not a mean: the distribution has a
long right tail (a slow decode, a GC pause) and a mean hides the p90 that
actually decides whether you hold frame rate.
"""
import resource
import statistics as st
import sys
import time
from pathlib import Path

import cv2

from . import detector, paths


def peak_rss_mb():
    """Peak resident set size of this process, in MB.

    ru_maxrss is in *bytes* on macOS and the BSDs and in *kilobytes* on Linux.
    Getting this wrong reports 1.8 GB as 1.8 MB, which reads as plausible.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def device_memory_mb(torch, dev):
    """What the accelerator holds, or None when there is nothing to ask."""
    d = str(dev)
    try:
        if d.startswith("mps"):
            return torch.mps.current_allocated_memory() / (1024 * 1024)
        if d.startswith("cuda") or d.isdigit():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return None                     # a probe must never fail the benchmark
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
    torch = __import__("torch")
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
    done = 0
    while done < args.warmup + args.frames:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:                       # short clip: loop rather than report fewer
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
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
    print(f"  memory      peak RSS {peak_rss_mb():7.0f} MB"
          + (f"   {str(dev)} {dmem:.0f} MB" if dmem is not None else ""))
    if weights.exists():
        print(f"  weights     {weights.stat().st_size / (1024 * 1024):.1f} MB on disk")
    return 0
