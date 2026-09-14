"""Register locally trained weights with Roboflow, so they can drive Label Assist.

Closing the loop: the model you just trained becomes the thing that pre-labels
your next batch of frames.

    cvkit upload-weights --version 3 --run runs/v3_baseline
    cvkit upload-weights --version 3 --run runs/v3_baseline --weights last.pt
"""
from pathlib import Path

from .. import config


def add_args(ap):
    config.add_roboflow_args(ap)
    ap.add_argument("--version", type=int, required=True,
                    help="dataset version number to attach the model to")
    ap.add_argument("--run", required=True, help="training run dir, e.g. runs/v3_baseline")
    ap.add_argument("--weights", default="best.pt", help="file inside <run>/weights/")
    ap.add_argument("--model-type", default="yolov8",
                    help="architecture string Roboflow expects, e.g. yolov8, yolo11")
    return ap


def run(args):
    run_dir = Path(args.run).resolve()
    wpath = run_dir / "weights" / args.weights
    if not wpath.exists():
        raise SystemExit(f"no weights at {wpath}")
    ws, pr = config.target(args)
    print(f"uploading {wpath} ({wpath.stat().st_size / 1e6:.1f} MB) "
          f"as {args.model_type} -> {ws}/{pr} v{args.version}")

    key = config.api_key(args)
    project = config.rf_project(args)
    # Unwrapped, an SDK failure here is a traceback carrying the key.
    try:
        version = project.version(args.version)
        version.deploy(
            model_type=args.model_type,
            model_path=str(run_dir),         # packaging writes its bundle here
            filename=f"weights/{args.weights}",
        )
    except Exception as e:
        msg = config.scrub(f"{type(e).__name__}: {e}", key)
    else:
        print("upload call returned; server-side conversion takes a few minutes")
        return 0
    # Outside the handler, so the original is not left on __context__.
    raise SystemExit(f"upload failed: {msg}")
