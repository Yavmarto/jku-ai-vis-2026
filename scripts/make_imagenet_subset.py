"""Cut a small ImageNet / ImageNet-C subset out of the big archives.

The AkidaNet notebooks (Phase 1 - No AI/akidanet-imagenetc-card.ipynb and -sim.ipynb) only need the first N images of
every class for each condition, so there is no need to unpack ~50 GB on the machine with the card.
Run this once where the archives are, then copy the small output folder to the card machine.

Inputs (Zenodo ImageNet-C record 2235448, plus the ImageNet validation tar from image-net.org):
  --c-dir     folder holding weather.tar (fog), digital.tar (contrast), noise.tar (gaussian_noise)
  --val-tar   ILSVRC2012_img_val.tar (flat: all 50000 validation images in one folder).
              Optional: without it only the corrupted images are cut out (no clean baseline).
Output (the layout the notebook expects):
  <out>/imagenet-c/<corruption>/<severity>/<wnid>/<file>.JPEG
  <out>/val/<wnid>/<file>.JPEG

Usage:
  python scripts/make_imagenet_subset.py --c-dir D:/dl --val-tar D:/dl/ILSVRC2012_img_val.tar \
      --out data/imagenet_subset --n-per-class 2
Then in the notebook: IMAGENET_C_DIR = "<out>/imagenet-c", IMAGENET_VAL_DIR = "<out>/val".
"""
import argparse
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

TAR_FOR = {"fog": "weather.tar", "contrast": "digital.tar", "gaussian_noise": "noise.tar"}
SEVERITIES = ("1", "2", "3", "4", "5")
MEMBER = re.compile(r"(?:^|/)(?P<corr>[a-z_]+)/(?P<sev>[1-5])/(?P<wnid>n\d{8})/(?P<name>[^/]+)$")


def members(tar_path):
    """Iterate over the file members of a tar without keeping them all in memory.

    Opened as a stream ("r|*"): the Zenodo ImageNet-C files are named .tar but are gzip-compressed,
    which cannot be seeked, so every archive is read once from start to end.
    """
    with tarfile.open(tar_path, "r|*") as tf:
        for m in tf:
            tf.members = []  # tarfile caches every member; a 22 GB archive has ~750k of them
            if m.isfile():
                yield tf, m


def choose_sample(tar_path, corruption, n_per_class):
    """First n file names (sorted) of every class for one condition: {wnid: [names]}."""
    names = defaultdict(list)
    for _, m in members(tar_path):
        g = MEMBER.search(m.name)
        if g and g["corr"] == corruption and g["sev"] == "1":
            names[g["wnid"]].append(g["name"])
    return {w: sorted(v)[:n_per_class] for w, v in sorted(names.items())}


def save(tf, m, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(tf.extractfile(m).read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--c-dir", required=True, type=Path)
    ap.add_argument("--val-tar", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-per-class", type=int, default=2)
    ap.add_argument("--corruptions", nargs="+", default=list(TAR_FOR), choices=list(TAR_FOR))
    args = ap.parse_args()

    for c in args.corruptions:
        if not (args.c_dir / TAR_FOR[c]).is_file():
            sys.exit(f"missing {args.c_dir / TAR_FOR[c]} (needed for {c})")
    if args.val_tar and not args.val_tar.is_file():
        sys.exit(f"missing {args.val_tar}")

    # the sample is chosen from the smallest archive, because choosing needs one full read of it
    first = min(args.corruptions, key=lambda c: (args.c_dir / TAR_FOR[c]).stat().st_size)
    print(f"choosing the sample from {TAR_FOR[first]} ({first}, severity 1) ...", flush=True)
    sample = choose_sample(args.c_dir / TAR_FOR[first], first, args.n_per_class)
    if len(sample) != 1000:
        sys.exit(f"expected 1000 class folders, found {len(sample)}: wrong archive layout?")
    wanted = {(w, n) for w, names in sample.items() for n in names}
    wnid_of = {n: w for w, n in wanted}
    print(f"{len(wanted)} images per condition", flush=True)

    c_root = args.out / "imagenet-c"
    for c in args.corruptions:
        print(f"extracting {c} from {TAR_FOR[c]} ...", flush=True)
        got = 0
        for tf, m in members(args.c_dir / TAR_FOR[c]):
            g = MEMBER.search(m.name)
            if g and g["corr"] == c and (g["wnid"], g["name"]) in wanted:
                save(tf, m, c_root / c / g["sev"] / g["wnid"] / g["name"])
                got += 1
        expect = len(wanted) * len(SEVERITIES)
        print(f"  {got}/{expect} files" + ("" if got == expect else "  <-- INCOMPLETE"), flush=True)

    if args.val_tar:
        print("extracting clean validation images ...", flush=True)
        got = 0
        for tf, m in members(args.val_tar):
            name = m.name.rsplit("/", 1)[-1]
            if name in wnid_of:
                save(tf, m, args.out / "val" / wnid_of[name] / name)
                got += 1
        print(f"  {got}/{len(wanted)} files" + ("" if got == len(wanted) else "  <-- INCOMPLETE"))
        print(f"done. IMAGENET_C_DIR = {c_root}   IMAGENET_VAL_DIR = {args.out / 'val'}")
    else:
        print("no --val-tar: skipped the clean images (the notebook then has no clean baseline).")
        print(f"done. IMAGENET_C_DIR = {c_root}   IMAGENET_VAL_DIR = None")


if __name__ == "__main__":
    main()
