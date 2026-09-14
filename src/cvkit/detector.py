"""Loading an Ultralytics model without contaminating the working directory.

Two traps this exists to close:

1. A bare weight name like "yolo26s.pt" is downloaded by Ultralytics into the
   *current* directory. Run a script from a repo root and you have just added
   a 120 MB AGPL-licensed checkpoint to that repo. We chdir into a dedicated
   weights directory for the duration of the load.

2. chdir breaks a *relative* --model path. So a path that exists is resolved
   to absolute before the chdir; a bare name is left alone so the download
   still happens.

3. Ultralytics resolves other downloads against utils.WEIGHTS_DIR, which ships
   *relative*, so they land wherever the process is standing long after the
   chdir is restored. load() rebinds it absolute.

4. A .pt is a pickle, and Ultralytics unpickles it unrestricted unless
   ULTRALYTICS_SAFE_LOAD is set -- which old versions ignore. load() sets it
   and safe_load_gap() checks it took.

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


# Restricted loading: the flag landed in ultralytics 8.4.67, its torch probe needs 2.6.
SAFE_LOAD_MIN = "'ultralytics>=8.4.67' 'torch>=2.6'"


def safe_load_gap():
    """Why restricted checkpoint loading is *not* in force, or None if it is.

    Setting the flag is not evidence it took: an old ultralytics ignores it,
    and torch_safe_load drops back to an unrestricted load -- silently -- when
    its torch probe fails. A pyproject floor cannot settle it either, since an
    existing venv holds whatever it holds, so ask the installed code.
    """
    try:
        from ultralytics.utils import SAFE_LOAD
    except ImportError:
        return "this ultralytics is older than 8.4.67 and ignores the flag"
    if not SAFE_LOAD:
        return "ultralytics read ULTRALYTICS_SAFE_LOAD as off"
    # _SafeLoad is private: report a gap only on positive evidence, so a
    # rename upstream cannot block a working install.
    try:
        from ultralytics.nn.tasks import _SafeLoad
    except ImportError:
        return None
    if not getattr(_SafeLoad, "SUPPORTED", True):
        return "this torch is too old for it"
    return None


def pin_weights_dir(ultralytics, wdir):
    """Point Ultralytics' WEIGHTS_DIR at an absolute path.

    It ships relative, and the two consumers that matter import it inside a
    function -- so they read it after load()'s chdir is restored and write into
    the user's project: CLIP on a world model's set_classes() (353 MB,
    observed) and yolo26n.pt for the AMP probe on train(). The chdir is still
    needed: YOLO("yolo11s.pt") fetches a bare name without consulting this.
    """
    ultralytics.utils.WEIGHTS_DIR = Path(wdir)


def load(model_name, weights_dir, device=None):
    """(YOLO model, device). See the module docstring for why this is fiddly."""
    candidate = Path(model_name)
    if candidate.exists():
        model_name = str(candidate.resolve())

    wdir = Path(weights_dir).resolve()          # absolute: see pin_weights_dir
    wdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(wdir))
    # Must precede the ultralytics import below: the flag is read once, into a
    # module constant. ULTRALYTICS_SAFE_LOAD=0 opts out of the check.
    flag = os.environ.get("ULTRALYTICS_SAFE_LOAD", "1").strip().lower()
    opted_out = flag not in ("1", "true", "yes", "on", "y", "t")   # env_bool's set
    os.environ.setdefault("ULTRALYTICS_SAFE_LOAD", "1")

    cwd = Path.cwd()
    os.chdir(wdir)
    try:
        ul = config.require("ultralytics", "detect")
        pin_weights_dir(ul, wdir)
        YOLO = ul.YOLO
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
