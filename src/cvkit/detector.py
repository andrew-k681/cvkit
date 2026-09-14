"""Loading an Ultralytics model without contaminating the working directory.

Two traps this exists to close:

1. A bare weight name like "yolo26s.pt" is downloaded by Ultralytics into the
   *current* directory. Run a script from a repo root and you have just added
   a 120 MB AGPL-licensed checkpoint to that repo. We chdir into a dedicated
   weights directory for the duration of the load.

2. chdir breaks a *relative* --model path. So a path that exists is resolved
   to absolute before the chdir; a bare name is left alone so the download
   still happens.

It is also the one place a checkpoint is unpickled, so it is where restricted
loading gets turned on -- and, just as importantly, where cvkit checks that it
actually took. See safe_load_gap().

Importing torch/ultralytics is deferred to call time: the frame, sheet and
export tools must stay usable in an install that has neither.
"""
import os
from pathlib import Path

from . import config


def pick_device(preferred=None):
    """Explicit choice, else the best accelerator torch reports."""
    if preferred:
        return preferred
    torch = config.require("torch", "detect")
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return "cpu"


# The versions that actually implement restricted loading. The flag landed in
# ultralytics 8.4.67 (8.4.66 does not have it), and the torch probe behind it
# moved from `safe_globals` to `get_unsafe_globals_in_checkpoint` by 8.4.150,
# which is torch 2.6.
SAFE_LOAD_MIN = "'ultralytics>=8.4.67' 'torch>=2.6'"


def safe_load_gap():
    """Why restricted checkpoint loading is *not* in force, or None if it is.

    Setting ULTRALYTICS_SAFE_LOAD is not enough to rely on, because both halves
    of the feature fail quietly. The flag does not exist before ultralytics
    8.4.67, where it is simply ignored. And `torch_safe_load` does
    `if safe_only and not _SafeLoad.SUPPORTED: safe_only = False` -- a silent
    fallback to an unrestricted load -- where SUPPORTED is a hasattr probe
    against torch.serialization.

    A floor in pyproject.toml does not settle it either: an existing venv holds
    whatever it holds. So ask the installed code, and let load() refuse rather
    than let a docstring promise something that is not happening.
    """
    try:
        from ultralytics.utils import SAFE_LOAD
    except ImportError:
        return "this ultralytics is older than 8.4.67 and ignores the flag"
    if not SAFE_LOAD:
        return "ultralytics read ULTRALYTICS_SAFE_LOAD as off"
    # _SafeLoad is private, so a rename upstream must not become a false alarm
    # that blocks a perfectly good install: only report a gap on positive
    # evidence that the torch probe failed.
    try:
        from ultralytics.nn.tasks import _SafeLoad
    except ImportError:
        return None
    if not getattr(_SafeLoad, "SUPPORTED", True):
        return "this torch is too old for it"
    return None


def load(model_name, weights_dir, device=None):
    """(YOLO model, device). See the module docstring for why this is fiddly."""
    candidate = Path(model_name)
    if candidate.exists():
        model_name = str(candidate.resolve())

    wdir = Path(weights_dir)
    wdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(wdir))
    # A .pt checkpoint is a pickle, and Ultralytics loads one through
    # torch.load with weights_only=False unless this is set -- so `--model
    # something-someone-sent-you.pt` is arbitrary code execution at load time.
    # Restricted loading rebuilds only the model classes on its allow-list,
    # which covers every checkpoint this toolkit trains or downloads.
    #
    # It has to be set before ultralytics is imported: the flag is read once,
    # into a module constant. That is why the require() below sits after it.
    opted_out = os.environ.get("ULTRALYTICS_SAFE_LOAD", "1").strip().lower() in (
        "", "0", "f", "false", "n", "no", "off")
    os.environ.setdefault("ULTRALYTICS_SAFE_LOAD", "1")

    cwd = Path.cwd()
    os.chdir(wdir)
    try:
        YOLO = config.require("ultralytics", "detect").YOLO
        gap = None if opted_out else safe_load_gap()
        if gap:
            raise SystemExit(
                f"refusing to load {model_name}: restricted checkpoint loading "
                f"is not in force ({gap}). A .pt file is a pickle, so loading "
                f"one runs whatever it contains.\n"
                f"    pip install -U {SAFE_LOAD_MIN}\n"
                f"or, if you trust this checkpoint, set ULTRALYTICS_SAFE_LOAD=0.")
        dev = pick_device(device)
        model = YOLO(model_name)
    finally:
        os.chdir(cwd)                       # restore even if the load raises
    return model, dev


def resolve_classes(names, spec):
    """'vehicle', 'person,vehicle' or '0,1' -> [ids]. None means every class."""
    if not spec:
        return None
    by_name = {v: k for k, v in names.items()}
    ids = []
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        if tok.lstrip("-").isdigit() and int(tok) in names:
            ids.append(int(tok))
        elif tok in by_name:
            ids.append(by_name[tok])
        else:
            raise SystemExit(f"unknown class {tok!r}; model has "
                             f"{sorted(by_name)} / {sorted(names)}")
    return sorted(set(ids)) or None


def add_model_args(ap, default_model="yolo11s.pt", default_imgsz=640):
    ap.add_argument("--model", default=default_model,
                    help="checkpoint path, or a name Ultralytics can download "
                         f"(default: {default_model})")
    ap.add_argument("--imgsz", type=int, default=default_imgsz)
    ap.add_argument("--device", default=None, help="mps / cuda / 0 / cpu (default: auto)")
    ap.add_argument("--weights-dir", default=None,
                    help="where downloaded checkpoints go (default: data/mining/weights)")
    return ap
