"""Find test images that are near-copies of training images, and re-score the model without them.

The dataset was collected from the web, and some photos ended up in more than one split
(the same picture saved twice under different file names). A copy of a training image in
the test split makes the test score look better than it should, so this script measures
how much that matters.

Usage:
    python scripts/check_duplicates.py --data "<dataset folder containing train/ valid/ test/>" --model best.pt

Needs: ultralytics, imagehash, pyyaml, pillow
"""
import argparse
import glob
import json
import os
import tempfile

import imagehash
import yaml
from PIL import Image
from ultralytics import YOLO


def phashes(folder):
    return {os.path.basename(p): imagehash.phash(Image.open(p).convert("RGB"))
            for p in sorted(glob.glob(os.path.join(folder, "*")))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root (has train/, valid/, test/ and data.yaml)")
    ap.add_argument("--model", default="best.pt")
    ap.add_argument("--threshold", type=int, default=8,
                    help="max perceptual-hash distance counted as a duplicate (0 = identical)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    train = phashes(os.path.join(args.data, "train", "images"))
    test = phashes(os.path.join(args.data, "test", "images"))

    dupes = {}
    for name, h in test.items():
        closest = min(train.items(), key=lambda kv: h - kv[1])
        if h - closest[1] <= args.threshold:
            dupes[name] = (closest[0], int(h - closest[1]))
    print(f"{len(dupes)} of {len(test)} test images have a near-copy in train (distance <= {args.threshold})")
    for t, (tr, d) in sorted(dupes.items()):
        print(f"  {t}  ~  {tr}  (distance {d})")

    # Build a test split without the duplicates (symlinks, nothing is copied) and evaluate on it.
    out = tempfile.mkdtemp(prefix="dedup_")
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(out, "test", sub))
    boxes = 0
    for name in test:
        if name in dupes:
            continue
        stem = os.path.splitext(name)[0]
        label = os.path.join(args.data, "test", "labels", stem + ".txt")
        os.symlink(os.path.join(args.data, "test", "images", name), os.path.join(out, "test", "images", name))
        os.symlink(label, os.path.join(out, "test", "labels", stem + ".txt"))
        with open(label) as f:
            boxes += sum(1 for line in f if line.strip())

    with open(os.path.join(args.data, "data.yaml")) as f:
        names = yaml.safe_load(f)["names"]
    cfg = os.path.join(out, "data.yaml")
    with open(cfg, "w") as f:
        yaml.safe_dump({"path": out, "train": "test/images", "val": "test/images", "test": "test/images",
                        "nc": len(names), "names": names}, f)

    m = YOLO(args.model).val(data=cfg, split="test", imgsz=640, batch=16, device=args.device,
                             plots=False, verbose=False)
    result = {
        "test_images_total": len(test),
        "duplicates_removed": len(dupes),
        "test_images_kept": len(test) - len(dupes),
        "boxes_kept": boxes,
        "phash_threshold": args.threshold,
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
        "duplicates": {t: tr for t, (tr, _) in sorted(dupes.items())},
    }
    print(json.dumps({k: v for k, v in result.items() if k != "duplicates"}, indent=2))
    with open("dedup_check.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Wrote dedup_check.json")


if __name__ == "__main__":
    main()
