"""Download source videos listed in a urls file.

Video only, no audio: nothing downstream listens, and the audio stream doubles
the download for nothing. A download archive is kept alongside the videos so
re-running only fetches what is new.

    cvkit download
    cvkit download --urls urls.txt --out data/raw
    cvkit download --cookies firefox        # if the site asks you to sign in

The urls file is *input*, not configuration. read_urls and build_command are
what stop a line of it becoming a yt-dlp option.
"""
import subprocess
import sys
from pathlib import Path

from .. import config, paths

# best mp4 video-only stream at 1080p or below, else any mp4
DEFAULT_FORMAT = "bv*[ext=mp4][height<=1080]/b[ext=mp4]"


def read_urls(path):
    """URLs from a urls file: one per line, blank lines and # comments skipped.

    A line starting with `-` is refused: yt-dlp would read it as an option, and
    `--exec=...` runs commands. build_command also puts `--` in front of the
    URLs -- this is the half that holds if that separator is ever dropped.
    """
    urls = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        u = line.strip()
        if not u or u.startswith("#"):
            continue
        if u.startswith("-"):
            raise SystemExit(
                f"{path}:{n}: a URL cannot start with '-' -- yt-dlp would read "
                f"this as an option rather than an address:\n    {u[:80]}")
        urls.append(u)
    return urls


def build_command(urls, out, fmt, template, cookies=None):
    """The yt-dlp argv. Split out of run() so the `--` separator is testable."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", fmt,
        "--write-info-json",
        "--no-overwrites",
        "--download-archive", str(out / "archive.txt"),
        "-o", str(out / template),
        "--ignore-errors",
    ]
    if cookies:
        cmd += ["--cookies-from-browser", cookies]
    # Everything after `--` is an address, whatever character it starts with.
    return cmd + ["--"] + list(urls)


def add_args(ap):
    paths.add_root_arg(ap)
    ap.add_argument("--urls", default="urls.txt",
                    help="file of URLs, one per line, # for comments (default: urls.txt)")
    ap.add_argument("--out", default=None, help="destination (default: data/raw)")
    ap.add_argument("--format", default=DEFAULT_FORMAT, help="yt-dlp format selector")
    ap.add_argument("--cookies", help="browser to take cookies from, e.g. firefox, chrome")
    ap.add_argument("--template", default="%(id)s.%(ext)s",
                    help="yt-dlp output template (default: %%(id)s.%%(ext)s)")
    return ap


def run(args):
    # yt-dlp is run as a subprocess, not imported, so check for it here --
    # otherwise a missing install surfaces as a bare "No module named yt_dlp"
    # from the child interpreter instead of naming the extra to install.
    config.require("yt_dlp", "download")

    ws = paths.workspace(args)
    out = paths.resolve(args.out, ws.raw)
    urls_file = Path(args.urls)
    if not urls_file.is_absolute():
        urls_file = ws.root / urls_file
    if not urls_file.exists():
        raise SystemExit(f"no urls file at {urls_file}")

    urls = read_urls(urls_file)
    if not urls:
        raise SystemExit(f"no URLs in {urls_file}")

    out.mkdir(parents=True, exist_ok=True)
    cmd = build_command(urls, out, args.format, args.template, args.cookies)
    print(f"downloading {len(urls)} videos -> {out}/")
    return subprocess.run(cmd).returncode
