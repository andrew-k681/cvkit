"""Unit tests for the pure logic: naming, parsing, and label maths.

Deliberately free of torch, the Roboflow SDK and the network, so they run in a
base install. The detector-driven paths are exercised by hand; what is tested
here is the arithmetic that silently corrupts a dataset when it is wrong.

Twelve need an extra anyway -- `fix_export` and `mining.sheets` carry a
module-level numpy/cv2 import by convention, and PyYAML backs the data.yaml
guard. Those skip rather than fail collection for everyone.
"""
import json
import sys
import types

import pytest

from cvkit import detector, naming, train
from cvkit.config import read_env_var, require, scrub
from cvkit.roboflow import search
from cvkit.video.download import build_command, read_urls
from cvkit.video.frames import parse_offsets

try:
    from cvkit.dataset.fix_export import polygon_to_bbox
    from cvkit.mining.sheets import parse_reject, subsample
except ImportError:                     # no images extra: numpy/cv2 absent
    polygon_to_bbox = parse_reject = subsample = None

needs_images = pytest.mark.skipif(
    polygon_to_bbox is None,
    reason="needs the images extra: pip install 'cvkit[images]'")

try:
    import yaml as _yaml
except ImportError:                     # PyYAML ships in the dev and detect extras
    _yaml = None

# Without it check_data_yaml exits on the missing import, not on what is being
# tested -- which two of these assert loosely enough to accept.
needs_yaml = pytest.mark.skipif(
    _yaml is None, reason="needs PyYAML: pip install -e '.[dev]'")


# ------------------------------------------------------------------ naming

@pytest.mark.parametrize("idx", [0, 1, 5, 42, 203, 9999, -1, -2, -16, -203])
def test_frame_name_round_trips(idx):
    name = naming.frame_name("clip01", idx)
    assert naming.split(name) == ("clip01", idx)


def test_negative_indices_keep_full_width():
    """The sign sits outside the padding, so -16 keeps four digits.

    A plain f"{idx:04d}" gives '-016' and loses one, which stops the name
    matching the rest of the dataset.
    """
    assert naming.frame_name("clip", -16) == "clip_mp4--0016.jpg"
    assert naming.frame_name("clip", 16) == "clip_mp4-0016.jpg"


def test_strips_roboflow_export_suffix():
    exported = "clip01_mp4--0016_jpg.rf.0123456789abcdef0123456789abcdef.jpg"
    assert naming.split(exported) == ("clip01", -16)


def test_video_id_may_contain_the_separator():
    """rpartition, not partition: the last separator is the real one."""
    assert naming.split("a_mp4-b_mp4-0007.jpg") == ("a_mp4-b", 7)


def test_custom_separator():
    assert naming.frame_name("clip", 7, sep="-frame") == "clip-frame0007.jpg"
    assert naming.split("clip-frame0007.jpg", sep="-frame") == ("clip", 7)


def test_unparseable_names():
    assert naming.try_split("not_a_frame.jpg") is None
    with pytest.raises(ValueError):
        naming.split("not_a_frame.jpg")


# ----------------------------------------------------------------- offsets

def test_offsets_from_flags_and_file(tmp_path):
    f = tmp_path / "offsets.json"
    f.write_text(json.dumps({"a": 1, "b": 2}))
    # an explicit --offset overrides the same key in the file
    assert parse_offsets(["b=9", "c=3"], str(f)) == {"a": 1, "b": 9, "c": 3}


def test_offsets_reject_malformed():
    with pytest.raises(SystemExit):
        parse_offsets(["nope"], None)


# --------------------------------------------------------------- rejection

@needs_images
def test_parse_reject():
    assert parse_reject("bg_002:5,17 bg_004:3") == {
        ("bg_002", 5), ("bg_002", 17), ("bg_004", 3)}
    assert parse_reject("") == set()
    assert parse_reject(None) == set()


@needs_images
def test_parse_reject_rejects_malformed():
    with pytest.raises(SystemExit):
        parse_reject("bg_002")


# --------------------------------------------------------------- subsample

@needs_images
def test_subsample_spreads_across_videos():
    frames = ([naming.frame_name("a", i) for i in range(100)]
              + [naming.frame_name("b", i) for i in range(100)])
    picked = subsample(frames, 20)
    assert len(picked) == 20
    by_video = {}
    for f in picked:
        by_video.setdefault(naming.split(f)[0], []).append(f)
    # both videos represented, neither dominating
    assert set(by_video) == {"a", "b"}
    assert all(len(v) >= 8 for v in by_video.values())


@needs_images
def test_subsample_spreads_along_the_timeline():
    frames = [naming.frame_name("a", i) for i in range(100)]
    picked = sorted(naming.split(f)[1] for f in subsample(frames, 10))
    assert picked[0] < 10 and picked[-1] > 80, picked


@needs_images
def test_subsample_caps_at_what_exists():
    frames = [naming.frame_name("a", i) for i in range(3)]
    assert len(subsample(frames, 50)) == 3


# ------------------------------------------------------------ polygon fix

@needs_images
def test_polygon_to_bbox():
    poly = "0 0.10 0.20 0.30 0.20 0.30 0.60 0.10 0.60".split()
    assert polygon_to_bbox(poly) == "0 0.200000 0.400000 0.200000 0.400000"


@needs_images
def test_polygon_to_bbox_keeps_the_class():
    poly = "7 0.0 0.0 1.0 1.0 0.5 0.5".split()
    assert polygon_to_bbox(poly).split()[0] == "7"


# ----------------------------------------------------------------- dotenv

def test_read_env_var_handles_quotes_and_export(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        '# a comment\n'
        'ROBOFLOW_API_KEY="quoted-key"\n'
        "export ROBOFLOW_WORKSPACE='ws'\n"
        "ROBOFLOW_PROJECT = spaced\n"
    )
    assert read_env_var(env, "ROBOFLOW_API_KEY") == "quoted-key"
    assert read_env_var(env, "ROBOFLOW_WORKSPACE") == "ws"
    assert read_env_var(env, "ROBOFLOW_PROJECT") == "spaced"
    assert read_env_var(env, "ABSENT") is None


def test_read_env_var_does_not_match_a_longer_name(tmp_path):
    """Reading KEY must not pick up MY_KEY's value."""
    env = tmp_path / ".env"
    env.write_text("MY_ROBOFLOW_API_KEY=wrong\nROBOFLOW_API_KEY=right\n")
    assert read_env_var(env, "ROBOFLOW_API_KEY") == "right"


def test_read_env_var_missing_file(tmp_path):
    assert read_env_var(tmp_path / "nope.env", "ANY") is None


# ------------------------------------------------- urls file -> yt-dlp argv
#
# yt-dlp has options that run commands (--exec). Both halves of the defence are
# tested: the parser refuses an option-shaped line, and the argv puts `--`
# in front of the URLs.

def test_read_urls_skips_blanks_and_comments(tmp_path):
    f = tmp_path / "urls.txt"
    f.write_text("https://a/1\n\n  # a comment\n  https://a/2  \n")
    assert read_urls(f) == ["https://a/1", "https://a/2"]


@pytest.mark.parametrize("line", [
    "--exec=touch /tmp/pwned",
    "-o/tmp/anywhere",
    "  --cookies-from-browser chrome",
])
def test_read_urls_refuses_an_option_shaped_line(tmp_path, line):
    f = tmp_path / "urls.txt"
    f.write_text(f"https://a/1\n{line}\n")
    with pytest.raises(SystemExit) as e:
        read_urls(f)
    assert "option" in str(e.value)


def test_build_command_separates_urls_from_options(tmp_path):
    cmd = build_command(["https://a/1", "https://a/2"], tmp_path,
                        "bv*", "%(id)s.%(ext)s")
    assert cmd[-3:] == ["--", "https://a/1", "https://a/2"]
    # nothing yt-dlp could read as an option may follow the separator
    assert not any(a.startswith("-") for a in cmd[cmd.index("--") + 1:])


def test_build_command_keeps_cookies_before_the_separator(tmp_path):
    cmd = build_command(["https://a/1"], tmp_path, "bv*", "%(id)s.%(ext)s",
                        cookies="firefox")
    assert cmd.index("--cookies-from-browser") < cmd.index("--")


# ------------------------------------------------------------- data.yaml
#
# Ultralytics executes a `download:` field whenever the val images are missing,
# so cvkit refuses the key before training.

@needs_yaml
def test_data_yaml_with_a_download_key_is_refused(tmp_path):
    y = tmp_path / "data.yaml"
    y.write_text("train: images/train\nval: /nope\nnc: 1\nnames: [a]\n"
                 "download: |\n  import os; os.system('id')\n")
    with pytest.raises(SystemExit) as e:
        train.check_data_yaml(y)
    assert "download" in str(e.value)


@needs_yaml
def test_download_key_is_refused_in_flow_style(tmp_path):
    """Parsed, not pattern-matched: `{download: ...}` must not slip past."""
    y = tmp_path / "data.yaml"
    y.write_text("{train: images/train, val: /nope, download: 'bash evil.sh'}\n")
    with pytest.raises(SystemExit):
        train.check_data_yaml(y)


@needs_yaml
def test_an_ordinary_export_yaml_passes(tmp_path):
    y = tmp_path / "data.yaml"
    y.write_text("train: ../train/images\nval: ../valid/images\n"
                 "test: ../test/images\nnc: 2\nnames: ['vehicle', 'person']\n")
    assert train.check_data_yaml(y)["nc"] == 2


@needs_yaml
def test_unreadable_yaml_is_a_clean_exit_not_a_traceback(tmp_path):
    y = tmp_path / "data.yaml"
    y.write_text("names: [unclosed\n")
    with pytest.raises(SystemExit):
        train.check_data_yaml(y)


# --------------------------------------------------- api key in error text
#
# The key rides in the query string, so requests names it in every error it
# raises, and an uncaught one reaches a traceback and a CI log.

KEY = "rf_SUPERSECRET123"


class _Boom:
    """Stands in for requests: raises the way a real failure does."""

    class _Response:
        url = f"https://api.roboflow.com/ws/pr/search?api_key={KEY}"

        def raise_for_status(self):
            raise RuntimeError(f"401 Client Error: Unauthorized for url: {self.url}")

    def post(self, url, **kw):
        return self._Response()

    get = post


def test_api_errors_do_not_carry_the_key():
    with pytest.raises(search.ApiError) as e:
        search.page(_Boom(), KEY, "ws", "pr", {}, 0)
    assert KEY not in str(e.value)
    assert "<redacted>" in str(e.value)


def test_api_error_does_not_chain_the_original_unscrubbed_exception():
    """`from None` only sets a flag; anything walking __context__ still finds
    the key. _call raises outside the handler, so nothing is attached."""
    with pytest.raises(search.ApiError) as e:
        search.page(_Boom(), KEY, "ws", "pr", {}, 0)
    assert e.value.__cause__ is None and e.value.__context__ is None


def test_api_error_is_catchable_by_the_per_item_loops():
    """Bulk operations count failures per item; SystemExit would abort them."""
    assert issubclass(search.ApiError, Exception)
    assert not issubclass(search.ApiError, SystemExit)


def test_api_error_reason_survives_the_per_item_truncation():
    """The loops print str(e)[:90], and one image URL fills that on its own."""
    long_url = f"{search.API}/my-workspace/my-project/images/68f3abc123def4567890abcd/tags"

    class _Boom:
        class _Response:
            def raise_for_status(self):
                raise RuntimeError(
                    f"401 Client Error: Unauthorized for url: {long_url}?api_key={KEY}")

        def post(self, url, **kw):
            return self._Response()

    with pytest.raises(search.ApiError) as e:
        search.set_tags(_Boom(), KEY, "my-workspace", "my-project",
                        "68f3abc123def4567890abcd", ["t"], "add")
    head = str(e.value)[:90]
    assert "401" in head and "Unauthorized" in head
    assert KEY not in str(e.value)


def test_scrub_blanks_the_key_anywhere_in_the_text():
    assert scrub(f"url: /ws/pr?api_key={KEY} (retries)", KEY) == \
        "url: /ws/pr?api_key=<redacted> (retries)"
    assert scrub("nothing to hide", KEY) == "nothing to hide"


# ------------------------------------------------- restricted model loading
#
# Setting ULTRALYTICS_SAFE_LOAD is not evidence it took -- an old ultralytics
# ignores it, an old torch silently defeats it -- so safe_load_gap asks.

def _fake_ultralytics(monkeypatch, *, flag=True, on=True, tasks=True, supported=True):
    pkg = types.ModuleType("ultralytics")
    utils = types.ModuleType("ultralytics.utils")
    if flag:
        utils.SAFE_LOAD = on
    mods = {"ultralytics": pkg, "ultralytics.utils": utils,
            "ultralytics.nn": types.ModuleType("ultralytics.nn")}
    if tasks:
        t = types.ModuleType("ultralytics.nn.tasks")
        t._SafeLoad = type("_SafeLoad", (), {"SUPPORTED": supported})
        mods["ultralytics.nn.tasks"] = t
    else:
        mods["ultralytics.nn.tasks"] = None      # None in sys.modules -> ImportError
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)


def test_no_gap_when_both_halves_are_in_force(monkeypatch):
    _fake_ultralytics(monkeypatch)
    assert detector.safe_load_gap() is None


def test_gap_when_ultralytics_predates_the_flag(monkeypatch):
    _fake_ultralytics(monkeypatch, flag=False)
    assert "8.4.67" in detector.safe_load_gap()


def test_gap_when_the_flag_is_read_as_off(monkeypatch):
    _fake_ultralytics(monkeypatch, on=False)
    assert "off" in detector.safe_load_gap()


def test_gap_when_torch_is_too_old(monkeypatch):
    """torch_safe_load falls back to an unrestricted load without saying so."""
    _fake_ultralytics(monkeypatch, supported=False)
    assert "torch" in detector.safe_load_gap()


def test_a_renamed_private_class_is_not_a_false_alarm(monkeypatch):
    """If it moves the public flag is still on; do not block a working install."""
    _fake_ultralytics(monkeypatch, tasks=False)
    assert detector.safe_load_gap() is None


# ------------------------------------------------------- install messages

def test_require_names_the_distribution_not_the_module():
    """`pip install pyyaml`, not `pip install yaml`."""
    with pytest.raises(SystemExit) as e:
        require("definitely_not_a_module", "detect", "PyYAML")
    assert "PyYAML" in str(e.value) and "cvkit[detect]" in str(e.value)


@needs_yaml
def test_a_directory_is_not_reported_as_corrupt_yaml(tmp_path):
    """Ultralytics accepts one here; cvkit does not, and should say which."""
    d = tmp_path / "my-export"
    d.mkdir()
    with pytest.raises(SystemExit) as e:
        train.check_data_yaml(d)
    assert "directory" in str(e.value)


# ------------------------------------------------------- review-video overlay

@needs_images
def test_draw_reads_a_prescan_tuple_and_hides_weak_keypoints():
    """Pins the cached tuple's shape and the threshold: a pose model invents
    the keypoints it cannot see."""
    import numpy as np

    from cvkit.mining.review_video import draw

    strong, weak = (30, 40, 0.9), (90, 40, 0.1)
    box = (10, 10, 110, 70, 0.8, 0, 7, [list(strong), list(weak)])
    view = draw(np.zeros((80, 120, 3), dtype=np.uint8), [box], 0, 1, 25.0,
                0.25, 0, True, False, "clip", {0: "person"}, "", kpt_conf=0.5)

    assert view[strong[1], strong[0]].any()
    assert not view[weak[1], weak[0]].any()


@needs_images
def test_event_ranges_are_clamped_and_reversed_ones_refused(tmp_path):
    """Parsed, not trusted: an out-of-range seek, a reversed range matches
    nothing."""
    from cvkit.mining.review_video import load_events

    p = tmp_path / "events.json"
    p.write_text('[{"start": -5, "end": 99999, "label": "hand_up"}]')
    assert load_events(p, 2090) == [(0, 2089, "hand_up")]

    # both ends, or a range wholly outside survives inverted and matches nothing
    p.write_text('[{"start": 5000, "end": 6000}]')
    assert load_events(p, 2090) == [(2089, 2089, "event")]
    p.write_text('[{"start": -50, "end": -10}]')
    assert load_events(p, 2090) == [(0, 0, "event")]

    p.write_text('[{"start": 90, "end": 10}]')
    with pytest.raises(SystemExit) as e:
        load_events(p, 2090)
    assert "after end" in str(e.value)


def test_weights_dir_is_pinned_absolute(monkeypatch, tmp_path):
    """It ships relative, so it resolves against the cwd after load()'s chdir
    is restored. Observed: a 353 MB ViT-B-32.pt in a project root."""
    import pathlib

    _fake_ultralytics(monkeypatch)
    ul = sys.modules["ultralytics"]
    ul.utils = sys.modules["ultralytics.utils"]
    ul.utils.WEIGHTS_DIR = pathlib.Path("weights")       # the shipped default

    detector.pin_weights_dir(ul, tmp_path / "w")

    assert ul.utils.WEIGHTS_DIR.is_absolute()
    assert ul.utils.WEIGHTS_DIR == tmp_path / "w"


@needs_images
def test_short_detection_rows_are_refused_at_load(tmp_path):
    """Indexed positionally downstream, so a short row must fail here."""
    from cvkit.mining.review_video import load_detections

    p = tmp_path / "d.json"
    p.write_text('{"names": {"0": "person"}, "frames": '
                 '[[[1, 2, 3, 4, 0.9, 0, 7, [[5, 6, 0.8]]]], [[1, 2, 3, 4, 0.9, 0]]]}')
    with pytest.raises(SystemExit) as e:
        load_detections(p, 100, 100)
    assert "frame 1" in str(e.value) and "6 fields" in str(e.value)

    p.write_text('{"frames": [[[1, 2, 3, 4, 0.9, 0, null, []]]]}')
    names, frames = load_detections(p, 100, 100)
    assert names == {0: "object"}                      # default when unnamed
    assert frames[0][0][6] is None and frames[0][0][7] == []


@needs_images
def test_detections_accept_a_bare_prescan_cache(tmp_path):
    """prescan writes a bare list; the docs promise it is accepted."""
    from cvkit.mining.review_video import load_detections

    p = tmp_path / "cache.json"
    p.write_text('[[[1, 2, 3, 4, 0.9, 0, 7, [[5, 6, 0.8]]]], []]')
    names, frames = load_detections(p, 100, 100)
    assert names == {0: "object"} and len(frames) == 2 and frames[1] == []


@needs_images
def test_detection_coordinates_are_clamped_into_the_frame(tmp_path):
    """OpenCV takes C ints: out of range raises OverflowError mid-render."""
    from cvkit.mining.review_video import load_detections

    p = tmp_path / "d.json"
    p.write_text('{"frames": [[[-9, -9, 10000000000000000000, 5, 0.9, 0, null, '
                 '[[-4, 99999999999999999999, 0.8]]]]]}')
    _names, frames = load_detections(p, 640, 480)
    x1, y1, x2, y2 = frames[0][0][:4]
    assert (x1, y1, x2, y2) == (0, 0, 640, 5)
    assert frames[0][0][7][0][:2] == [0, 480]


@needs_images
def test_unusable_detection_values_are_refused(tmp_path):
    """json.loads accepts NaN and Infinity; int() of either escapes the clamp."""
    from cvkit.mining.review_video import load_detections

    p = tmp_path / "d.json"
    for bad in ('NaN', 'Infinity', '"x"'):
        p.write_text('{"frames": [[[%s, 2, 3, 4, 0.9, 0, null, []]]]}' % bad)
        with pytest.raises(SystemExit) as e:
            load_detections(p, 640, 480)
        assert "frame 0" in str(e.value)


@needs_images
def test_a_detections_file_that_is_not_frames_is_refused(tmp_path):
    from cvkit.mining.review_video import load_detections

    p = tmp_path / "d.json"
    p.write_text('{"names": {"0": "person"}}')
    with pytest.raises(SystemExit) as e:
        load_detections(p, 100, 100)
    assert "frames" in str(e.value)


@needs_images
def test_peak_rss_unit_differs_by_platform(monkeypatch):
    """bytes on macOS/BSD, kilobytes on Linux: 1.8 GB reads as 1.8 MB."""
    import resource

    from cvkit import bench

    monkeypatch.setattr(resource, "getrusage",
                        lambda _who: types.SimpleNamespace(ru_maxrss=2 * 1024 * 1024))
    monkeypatch.setattr(bench.sys, "platform", "darwin")
    assert bench.peak_rss_mb() == 2                 # bytes -> 2 MB
    monkeypatch.setattr(bench.sys, "platform", "linux")
    assert bench.peak_rss_mb() == 2048              # kilobytes -> 2 GB
