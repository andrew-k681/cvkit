"""Where cvkit expects to find things, all hanging off one project root.

The root is the current working directory unless you say otherwise, so the
same install serves every project you point it at:

    cd ~/work/site-survey && cvkit frames   # ~/work/site-survey/data/frames
    cd ~/work/traffic && cvkit frames       # ~/work/traffic/data/frames

Every command also takes explicit --<dir> overrides; this only supplies the
defaults.
"""
import os
from pathlib import Path


class Workspace:
    """Default directory layout under a project root."""

    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("CVKIT_ROOT") or Path.cwd()).resolve()

    def __repr__(self):
        return f"Workspace({self.root})"

    @property
    def data(self):
        return self.root / "data"

    @property
    def raw(self):
        """Downloaded source videos."""
        return self.data / "raw"

    @property
    def frames(self):
        """Extracted frames, one subdirectory per video."""
        return self.data / "frames"

    @property
    def mining(self):
        """Detector scores, contact sheets, verdict files, track caches."""
        return self.data / "mining"

    @property
    def weights(self):
        """Where downloaded checkpoints land.

        Ultralytics fetches a bare weight name into the *current directory*,
        so left alone it drops AGPL-licensed .pt files into whatever repo you
        happened to be standing in. Everything that loads a model chdirs here
        first. Keep it outside version control.
        """
        return self.mining / "weights"

    @property
    def trackcache(self):
        return self.mining / "trackcache"

    @property
    def runs(self):
        """Training output."""
        return self.root / "runs"


def add_root_arg(ap):
    ap.add_argument("--root", default=None,
                    help="project root for default paths (default: $CVKIT_ROOT or cwd)")
    return ap


def workspace(args):
    return Workspace(getattr(args, "root", None))


def resolve(override, default):
    """An explicit --dir wins; otherwise use the workspace default."""
    return Path(override) if override else default
