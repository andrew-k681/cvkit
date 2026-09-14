"""The `cvkit` command.

Every subcommand lives in its own module exposing `add_args(parser)` and
`run(args)`. The modules are imported lazily, one per invocation, so `cvkit
frames` never pays for (or requires) torch, and `cvkit sheets` never requires
the Roboflow SDK.
"""
import argparse
import importlib
import sys
import textwrap

from . import __version__

# name -> (module, add_args attr, run attr, one-line help, extra needed)
# The last field is the optional-dependency group this command needs, or None
# if it runs on the base install. It is what the ImportError handler quotes,
# so the user is told the one extra to install rather than a list to guess from.
COMMANDS = {
    # video in, frames out
    "download": ("cvkit.video.download", "add_args", "run",
                 "fetch source videos listed in a urls file", "download"),
    "frames": ("cvkit.video.frames", "add_args", "run",
               "extract frames from every video in a directory", None),

    # a detector proposes, you dispose
    "mine": ("cvkit.mining.score", "add_args", "run",
             "score frames with a detector to shortlist candidates", "detect"),
    "sheets": ("cvkit.mining.sheets", "add_sheets_args", "run_sheets",
               "build contact sheets from mining scores", "images"),
    "collect": ("cvkit.mining.sheets", "add_collect_args", "run_collect",
                "copy out the frames that survived sheet review", "images"),
    "verify": ("cvkit.mining.sheets", "add_verify_args", "run_verify",
               "re-render collected frames large enough to judge", "images"),
    "record": ("cvkit.mining.sheets", "add_record_args", "run_record",
               "record verify-sheet verdicts so they are never re-reviewed", "images"),
    "review-video": ("cvkit.mining.review_video", "add_args", "run",
                     "step through a video with detections overlaid, banking frames",
                     "detect"),

    "bench": ("cvkit.bench", "add_args", "run",
              "time a model on your own footage: latency, throughput, memory",
              "detect"),

    # repair and train
    "fix-export": ("cvkit.dataset.fix_export", "add_args", "run",
                   "repair a YOLO export: polygon rows, cross-split duplicates",
                   "images"),
    "train": ("cvkit.train", "add_args", "run",
              "fine-tune a detector on a YOLO-format dataset", "detect"),

    # Roboflow
    "upload-images": ("cvkit.roboflow.upload_images", "add_args", "run",
                      "upload images into a Roboflow batch, split-aware", "roboflow"),
    "mark-null": ("cvkit.roboflow.mark_null", "add_args", "run",
                  "give background images the empty annotation they need", "roboflow"),
    "tag": ("cvkit.roboflow.tag", "add_args", "run",
            "tag dataset images that still need annotation work",
            "roboflow-api"),
    "upload-weights": ("cvkit.roboflow.upload_weights", "add_args", "run",
                       "register trained weights with a Roboflow version", "roboflow"),
}

EPILOG = textwrap.dedent("""\
    a typical loop:

      cvkit download                          # urls.txt -> data/raw
      cvkit frames                            # data/raw -> data/frames
      cvkit mine --export data/my-export      # score what is not already labelled
      cvkit sheets                            # contact sheets of the empty ones
      cvkit collect --reject "bg_002:5,17"    # drop the tiles that were not empty
      cvkit verify && cvkit record --bad "3"  # second look, at a usable size
      cvkit upload-images data/backgrounds --batch round2 --tags verified
      cvkit mark-null --tag verified          # make them count as backgrounds
      cvkit train --data data/my-export/data.yaml
      cvkit upload-weights --version 3 --run runs/baseline

    paths default to ./data and ./runs; set CVKIT_ROOT or pass --root to point
    them elsewhere. Credentials come from the environment or a .env file --
    see .env.example.
""")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="cvkit",
        description="Build object-detection datasets from video: source frames, "
                    "mine candidates with a detector, review them by hand, and "
                    "move the result into Roboflow.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--version", action="version", version=f"cvkit {__version__}")
    # Listed, not parsed. Dispatch happens in main() so each command's own
    # parser is built only after its module imports successfully -- most
    # commands need dependencies the others do not have installed, and
    # building every subparser up front would require importing all of them.
    sub = ap.add_subparsers(dest="command", metavar="<command>")
    for name, (_mod, _add, _run, help_text, _extra) in COMMANDS.items():
        sub.add_parser(name, help=help_text, add_help=False)
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()

    if not argv or argv[0] in ("-h", "--help"):
        ap.print_help()
        return 0
    if argv[0] in ("--version", "-V"):
        print(f"cvkit {__version__}")
        return 0

    name = argv[0]
    if name not in COMMANDS:
        ap.print_help(sys.stderr)
        print(f"\nunknown command {name!r}", file=sys.stderr)
        return 2

    mod_name, add_attr, run_attr, help_text, extra = COMMANDS[name]
    try:
        module = importlib.import_module(mod_name)
    except ImportError as e:
        hint = (f"    pip install 'cvkit[{extra}]'" if extra
                else "    pip install 'cvkit[all]'")
        print(f"`cvkit {name}` needs a dependency that is not installed: {e}\n{hint}",
              file=sys.stderr)
        return 1

    cmd = argparse.ArgumentParser(
        prog=f"cvkit {name}",
        description=(module.__doc__ or help_text),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    getattr(module, add_attr)(cmd)
    args = cmd.parse_args(argv[1:])
    return getattr(module, run_attr)(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
