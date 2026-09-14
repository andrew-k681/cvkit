"""Fine-tune an Ultralytics detector on a YOLO-format dataset.

Two defaults worth knowing about:

  * A fixed seed. Evaluation splits on a hand-built dataset are usually small
    -- a hundred images is normal -- so run-to-run noise is comparable to a
    real improvement. Compare runs at the same seed or not at all.

  * Weights are read from and downloaded into --weights-dir, so checkpoints
    never land in your repo root. Ultralytics fetches a bare weight name into
    the current directory; left alone that quietly commits a hundred megabytes
    of AGPL-licensed model to whatever project you were standing in.

Model licensing is yours to check. Ultralytics YOLO checkpoints are AGPL-3.0:
fine for research, a real obligation if you ship the result.

    cvkit train --data data/my-export/data.yaml
    cvkit train --data data/my-export/data.yaml --model yolo11n.pt --epochs 50 --imgsz 960

A data.yaml is a file you got from somewhere -- an export, a collaborator -- so
it is input, not configuration. check_data_yaml is what stops it running.
"""
from pathlib import Path

from . import config, detector, paths


def check_data_yaml(path):
    """Refuse a data.yaml that carries a `download:` key.

    Ultralytics *runs* that field: `bash ...` goes to subprocess, anything else
    goes to exec() (check_det_dataset, ultralytics/data/utils.py). It fires
    whenever the `val:` images are missing -- and a file written to attack you
    sets both keys, so neither condition is a barrier.

    Ultralytics' own stock dataset yamls (coco8.yaml and friends) use the key
    legitimately to fetch themselves. cvkit trains on an export you already
    have on disk, so refusing it here costs nothing that this toolkit does.
    """
    yaml = config.require("yaml", "detect", "PyYAML")
    # Ultralytics also accepts a directory or a zip here and finds the yaml
    # inside; this refuses both rather than guess, so say which, or the message
    # reads as though the file were corrupt.
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raise SystemExit(
            f"{path} is not a readable data.yaml -- point --data at the "
            f"data.yaml file itself, not a directory or an archive.")
    try:
        data = yaml.safe_load(text)
    except Exception as e:
        raise SystemExit(f"{path} is not valid YAML: {type(e).__name__} {e}")
    if isinstance(data, dict) and "download" in data:
        raise SystemExit(
            f"refusing {path}: it carries a `download:` key, which Ultralytics "
            f"executes as a shell or Python script when the val images are "
            f"missing. Delete the key if you trust the file, or point --data at "
            f"an export you produced yourself.")
    return data


def add_args(ap):
    paths.add_root_arg(ap)
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--model", default="yolo11s.pt",
                    help="starting checkpoint, or a name Ultralytics can download")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=25, help="early-stopping patience")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="mps / cuda / 0 / cpu (default: auto)")
    ap.add_argument("--name", default="baseline", help="run name under runs/")
    ap.add_argument("--project", default=None, help="run directory (default: runs/)")
    ap.add_argument("--weights-dir", default=None,
                    help="where downloaded checkpoints go (default: data/mining/weights)")
    ap.add_argument("--optimizer", default="auto")
    ap.add_argument("--no-plots", dest="plots", action="store_false")
    return ap


def run(args):
    ws = paths.workspace(args)
    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"no data.yaml at {data}")
    check_data_yaml(data)

    model, device = detector.load(args.model,
                                  paths.resolve(args.weights_dir, ws.weights),
                                  args.device)
    project = paths.resolve(args.project, ws.runs)
    print(f"{args.model} on {device} | imgsz={args.imgsz} batch={args.batch} "
          f"epochs={args.epochs} seed={args.seed}", flush=True)

    model.train(
        data=str(data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        seed=args.seed,
        pretrained=True,
        optimizer=args.optimizer,
        project=str(project),
        name=args.name,
        exist_ok=True,
        plots=args.plots,
    )
    print(f"\nrun written to {project / args.name}")
    return 0
