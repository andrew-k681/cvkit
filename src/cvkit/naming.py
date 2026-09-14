"""Frame filenames.

cvkit names an extracted frame `<video_id><sep><index>.jpg`, defaulting to the
`_mp4-` separator Roboflow produces when you upload a video and let it split
the frames. Keeping the same shape means a frame extracted locally and the
same frame downloaded inside an export are recognisably the same image, which
is what lets the mining step exclude frames you have already labelled.

Override the separator with --frame-sep if your dataset uses another one.
"""
import re

DEFAULT_SEP = "_mp4-"

# Roboflow appends this to every exported image: <stem>_jpg.rf.<32 hex>.jpg
RF_SUFFIX = re.compile(r"_jpg\.rf\.[0-9a-f]+\.jpg$", re.IGNORECASE)


def strip_export_suffix(filename):
    """`clip_mp4-0042_jpg.rf.ab12...jpg` -> `clip_mp4-0042`."""
    return RF_SUFFIX.sub("", filename)


def split(name, sep=DEFAULT_SEP):
    """`clip_mp4-0042.jpg` -> ('clip', 42). Raises ValueError if it does not fit."""
    stem = strip_export_suffix(name)
    stem = stem[:-4] if stem.lower().endswith(".jpg") else stem
    if sep not in stem:
        raise ValueError(f"{name!r} does not contain the frame separator {sep!r}")
    video, _, idx = stem.rpartition(sep)
    return video, int(idx)


def try_split(name, sep=DEFAULT_SEP):
    """split(), but None instead of an exception -- for filtering mixed dirs."""
    try:
        return split(name, sep)
    except ValueError:
        return None


def frame_name(video, idx, sep=DEFAULT_SEP, pad=4, ext=".jpg"):
    """('clip', 5) -> 'clip_mp4-0005.jpg'; ('clip', -16) -> 'clip_mp4--0016.jpg'.

    The sign sits outside the padding, so a negative index keeps all `pad`
    digits instead of losing one to the minus. Negative indices are normal --
    they are what an offset produces for frames before the point the platform
    started numbering from.
    """
    sign = "-" if idx < 0 else ""
    return f"{video}{sep}{sign}{abs(idx):0{pad}d}{ext}"


def add_sep_arg(ap):
    ap.add_argument("--frame-sep", default=DEFAULT_SEP,
                    help=f"separator between video id and frame index "
                         f"(default: {DEFAULT_SEP!r})")
    return ap
