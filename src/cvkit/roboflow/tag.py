"""Tag the images that still need work, so the UI can filter to them.

Tag them here, then search `tag:<name>` on the project's Images page and work
through the list. Re-run to refresh: the tag is a snapshot, not a live view.

    # everything that carries an annotation
    cvkit tag --tag reannotate --dry-run
    cvkit tag --tag reannotate

    # only what is still a hand-drawn box, dropping the tag off finished ones
    cvkit tag --tag reannotate --only-boxes

Background (null) images are left alone -- they have nothing to re-annotate.
"""
from .. import config
from . import search


def add_args(ap):
    config.add_roboflow_args(ap)
    ap.add_argument("--tag", default="reannotate", help="tag to apply")
    ap.add_argument("--split", default=None, help="limit to one split")
    ap.add_argument("--class", dest="klass", default=None,
                    help="only images containing this class")
    ap.add_argument("--only-boxes", action="store_true",
                    help="tag only images still holding a hand-drawn box, and "
                         "REMOVE the tag from ones whose objects are all "
                         "polygons -- so the tag means 'still to fix'")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def run(args):
    key = config.api_key(args)
    ws, pr = config.target(args)
    requests = config.require("requests", "roboflow")

    body = {"in_dataset": True, "fields": ["id", "name", "split", "annotations"]}
    if args.split:
        body["split"] = args.split
    if args.klass:
        body["class_name"] = args.klass
    images = search.all_images(requests, key, ws, pr, body)

    # The box count lives in `annotations` -- {"count": N, "classes": {...}}.
    # `labels` exists too but comes back empty for every image; reading it is
    # how you convince yourself a fully annotated project has no annotations.
    boxed = [im for im in images if (im.get("annotations") or {}).get("count")]
    empty = len(images) - len(boxed)
    by_split, by_class = {}, {}
    for im in boxed:
        s = im.get("split", "?")
        by_split[s] = by_split.get(s, 0) + 1
        for c, n in im["annotations"].get("classes", {}).items():
            by_class[c] = by_class.get(c, 0) + n
    print(f"in dataset: {len(images)} images -- {len(boxed)} annotated, {empty} background")
    print(f"by split: {by_split}")
    print(f"objects by class: {by_class}")

    drop = []
    if args.only_boxes:
        # An assisted-labelling tool writes {"type": "polygon", "points": [...]};
        # a hand-drawn box has neither key. That geometry is the only reliable
        # marker of which tool made an object -- there is no author field, and
        # `lastSet` just records the most recent edit, which says nothing about
        # how the object was drawn.
        todo_, done_ = [], []
        for i, im in enumerate(boxed, 1):
            a = search.annotation(requests, key, ws, pr, im["id"])
            objs = a.get("boxes", [])
            raw = [o for o in objs if o.get("type") != "polygon" and not o.get("points")]
            im["_mix"] = (len(raw), len(objs))
            (todo_ if raw else done_).append(im)
            if i % 100 == 0 or i == len(boxed):
                print(f"  reading geometry {i}/{len(boxed)}", flush=True)
        mixed = sum(1 for im in todo_ if im["_mix"][0] < im["_mix"][1])
        print(f"polygons only (done): {len(done_)}")
        print(f"still holding a plain box: {len(todo_)}  (of which mixed: {mixed})")
        boxed, drop = todo_, done_

    if args.dry_run or (not boxed and not drop):
        for im in boxed[:5]:
            print(f"   {str(im.get('name'))[:46]:<48} {str(im.get('split')):<6} "
                  f"{im['annotations']['classes']}")
        print(f"\nnothing changed (tag {args.tag!r})")
        return 0

    already = {im["id"] for im in
               search.all_images(requests, key, ws, pr,
                                 {"tag": args.tag, "fields": ["id"]}, quiet=True)}
    todo = [im for im in boxed if im["id"] not in already]
    print(f"already tagged: {len(already)}; left to tag: {len(todo)}")

    ok = fail = 0
    for i, im in enumerate(todo, 1):
        try:
            search.set_tags(requests, key, ws, pr, im["id"], [args.tag], "add")
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  FAILED {im['name']}: {type(e).__name__} {str(e)[:90]}")
        if i % 50 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  ok={ok} failed={fail}", flush=True)

    if drop:
        rm = 0
        for im in drop:
            if im["id"] not in already:
                continue
            try:
                search.set_tags(requests, key, ws, pr, im["id"], [args.tag], "remove")
                rm += 1
            except Exception as e:
                print(f"  FAILED untag {im['name']}: {type(e).__name__} {str(e)[:90]}")
        print(f"untagged {rm} already-fixed image(s)")

    final = search.all_images(requests, key, ws, pr,
                              {"tag": args.tag, "fields": ["id"]}, quiet=True)
    print(f"\ntagged this run: {ok}, failed: {fail}")
    print(f"total carrying {args.tag!r}: {len(final)} of {len(boxed)} candidates")
    if len(final) < len(boxed):
        print("  run it again -- search pagination is unstable, it will pick up the rest")
    return 1 if fail and not ok else 0
