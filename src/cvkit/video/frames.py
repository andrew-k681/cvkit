"""Extract frames from every video in a directory, one subdirectory per video.

Frames are named `<video_id><sep><index>.jpg` so they line up with the names
an annotation platform gives frames it split out of the same video -- see
cvkit.naming for why that matters.

Index offsets
-------------
If a platform already holds frames from these videos, its numbering probably
does not start where a fresh extraction does (it may have trimmed an intro, or
sampled from a re-encode). The mapping is

    second_in_video = dataset_index + offset[video_id]

Recover an offset once, by eye: extract at the same fps, then find which local
frame matches dataset frame 0 for that video. Record it and you never think
about it again:

    cvkit frames --offset clip01=15 --offset clip02=26
    cvkit frames --offsets offsets.json          # {"clip01": 15, ...}

Videos with no recorded offset start at 0.

    cvkit frames
    cvkit frames --fps 2 --force
"""
import json
import subprocess
from pathlib import Path

from .. import naming, paths


def parse_offsets(pairs, offsets_file):
    """--offset id=N (repeatable) merged over an --offsets json file."""
    out = {}
    if offsets_file:
        data = json.loads(Path(offsets_file).read_text(encoding="utf-8"))
        out.update({str(k): int(v) for k, v in data.items()})
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--offset expects id=seconds, got {pair!r}")
        vid, _, val = pair.partition("=")
        out[vid.strip()] = int(val)
    return out


def extract(video, out_root, fps, offset, sep, pad, force, quality=2):
    """Extract one video. Returns the number of frames written."""
    vid = video.stem
    dest = out_root / vid
    if dest.exists() and any(dest.glob("*.jpg")) and not force:
        n = len(list(dest.glob("*.jpg")))
        print(f"{vid:14} skip ({n} frames already)")
        return n
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.jpg"):
        old.unlink()

    # ffmpeg writes into a scratch dir first: it numbers sequentially from 1
    # and cannot express the offset, so renaming is a second pass.
    tmp = dest / "_tmp"
    tmp.mkdir(exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps}",
             "-q:v", str(quality), str(tmp / "f_%05d.jpg")],
            check=True,
        )
        n = 0
        for f in sorted(tmp.glob("*.jpg")):
            # ffmpeg numbers its output from 1, so at the default fps=1 the
            # sample index is the second in the video. At any other rate it is
            # just the nth sample, which is still a stable, sortable key.
            seq = int(f.stem.split("_")[1])
            f.rename(dest / naming.frame_name(vid, seq - offset, sep, pad))
            n += 1
    finally:
        for leftover in tmp.glob("*"):
            leftover.unlink()
        tmp.rmdir()
    print(f"{vid:14} {n:4} frames  (offset {offset:+d})")
    return n


def add_args(ap):
    paths.add_root_arg(ap)
    naming.add_sep_arg(ap)
    ap.add_argument("--videos", default=None, help="source directory (default: data/raw)")
    ap.add_argument("--out", default=None, help="destination (default: data/frames)")
    ap.add_argument("--fps", type=float, default=1.0, help="sampling rate (default: 1)")
    ap.add_argument("--ext", default="mp4", help="video extension to look for (default: mp4)")
    ap.add_argument("--pad", type=int, default=4, help="digits in the frame index (default: 4)")
    ap.add_argument("--offset", action="append", metavar="ID=SECONDS",
                    help="index offset for one video; repeatable")
    ap.add_argument("--offsets", default=None, help="json file of {video_id: offset}")
    ap.add_argument("--quality", type=int, default=2, help="ffmpeg -q:v, 2 is near-lossless")
    ap.add_argument("--force", action="store_true", help="re-extract even if frames exist")
    return ap


def run(args):
    ws = paths.workspace(args)
    src = paths.resolve(args.videos, ws.raw)
    out = paths.resolve(args.out, ws.frames)
    offsets = parse_offsets(args.offset, args.offsets)

    videos = sorted(src.glob(f"*.{args.ext.lstrip('.')}"))
    if not videos:
        raise SystemExit(f"no *.{args.ext} in {src}")
    out.mkdir(parents=True, exist_ok=True)
    total = sum(extract(v, out, args.fps, offsets.get(v.stem, 0),
                        args.frame_sep, args.pad, args.force, args.quality)
                for v in videos)
    print(f"\n{len(videos)} videos -> {total} frames in {out}")
    return 0
