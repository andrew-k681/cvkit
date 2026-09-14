"""Mark images as null (deliberate background) so they reach a dataset version.

An uploaded image with no annotation stays "unannotated" and never reaches a
dataset version -- clicking Add to Dataset is not enough. It needs an explicit
*empty* annotation, which is what records "this frame deliberately contains
nothing". Roboflow rejects an empty YOLO .txt as an unrecognised format, so
this writes Pascal VOC with zero <object> elements.

Background images are not filler: a detector that has never seen an empty
scene will confidently find your class in one.

    cvkit mark-null --tag round2 --dry-run
    cvkit mark-null --tag round2
"""
import tempfile
from pathlib import Path

from .. import config
from . import search

VOC = ('<annotation><folder></folder><filename>{name}</filename>'
       '<size><width>{w}</width><height>{h}</height><depth>3</depth></size>'
       '<segmented>0</segmented></annotation>')


def add_args(ap):
    config.add_roboflow_args(ap)
    ap.add_argument("--tag", required=True, help="only images carrying this tag")
    ap.add_argument("--width", type=int, default=1920, help="source image width")
    ap.add_argument("--height", type=int, default=1080, help="source image height")
    ap.add_argument("--in-dataset", action="store_true",
                    help="also consider images already in the dataset")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def run(args):
    key = config.api_key(args)
    ws, pr = config.target(args)
    requests = config.require("requests", "roboflow")

    body = {"tag": args.tag, "fields": ["id", "name"]}
    if not args.in_dataset:
        body["in_dataset"] = False
    found = search.all_images(requests, key, ws, pr, body, quiet=True)
    print(f"tag={args.tag!r}"
          f"{'' if args.in_dataset else ', not in dataset'}: {len(found)} images")
    if args.dry_run or not found:
        for f in found[:5]:
            print(f"   {f['name']}")
        if args.dry_run:
            print("(dry run -- nothing written)")
        return 0

    project = config.rf_project(args)
    ok = fail = 0
    with tempfile.TemporaryDirectory() as td:
        for i, img in enumerate(found, 1):
            xml = Path(td) / (Path(img["name"]).stem + ".xml")
            xml.write_text(VOC.format(name=img["name"], w=args.width, h=args.height))
            try:
                project.save_annotation(annotation_path=str(xml), image_id=img["id"],
                                        annotation_overwrite=True, num_retry_uploads=2)
                ok += 1
            except Exception as e:
                fail += 1
                # scrub: the SDK builds `...?api_key=<key>` URLs, and a
                # connection error names the URL it was trying.
                print(f"  FAILED {img['name']}: {type(e).__name__} "
                      f"{config.scrub(str(e), key)[:90]}")
            if i % 10 == 0 or i == len(found):
                print(f"  {i}/{len(found)}  ok={ok} failed={fail}", flush=True)
    print(f"\nmarked null: {ok}, failed: {fail}")
    return 1 if fail and not ok else 0
