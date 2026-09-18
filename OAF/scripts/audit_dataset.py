"""STEP 4 - Complete per-image dataset audit (read-only).

Walks a dataset root and records, for every image, the fields required by the
AETHER-OA X-ray data-audit spec.  Images are never modified.

Usage
-----
    python scripts/audit_dataset.py --source kaggle \
        --input-dir kaggle --out reports/dataset_audit_kaggle.csv

    # OAI repo ships metadata + code only (no pixel data) -> metadata audit:
    python scripts/audit_dataset.py --source oai --oai-summary \
        OAI-KL-Grade-Classification/data/OAI_summary.csv \
        --out reports/dataset_audit_oai.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    KL_CLASS_NAMES,
    get_logger,
    iter_images,
    parse_knee_filename,
    rel_to_root,
    sha256_file,
)

LOG = get_logger("audit")
Image.MAX_IMAGE_PIXELS = None            # these are medical scans; disable decompression-bomb cap
ImageFile.LOAD_TRUNCATED_IMAGES = False  # we WANT truncated files to raise so we can flag them

try:
    import imagehash
except ImportError:  # pragma: no cover
    imagehash = None
    LOG.warning("imagehash not installed - perceptual_hash column will be empty")


AUDIT_COLUMNS = [
    "dataset_source", "file_path", "filename", "class", "class_name",
    "patient_id", "knee_side", "original_split",
    "width", "height", "channels", "dtype", "file_size", "format",
    "is_readable", "is_corrupt",
    "mean_pixel", "std_pixel", "min_pixel", "max_pixel",
    "aspect_ratio", "unique_hash", "perceptual_hash",
]


def _class_from_path(path: Path, input_dir: Path) -> tuple[str, str]:
    """Infer (class, original_split) from the directory layout .../<split>/<class>/<file>."""
    parts = path.relative_to(input_dir).parts
    cls, split = "", ""
    # layout: <split>/<class>/<file>  OR  <class>/<file>
    if len(parts) >= 3:
        split, cls = parts[0], parts[1]
    elif len(parts) == 2:
        cls = parts[0]
    return cls, split


def audit_image(path: Path, source: str, input_dir: Path) -> dict:
    cls, split = _class_from_path(path, input_dir)
    knee = parse_knee_filename(path.stem)
    row = {
        "dataset_source": source,
        "file_path": rel_to_root(path),
        "filename": path.name,
        "class": cls,
        "class_name": KL_CLASS_NAMES.get(int(cls), "") if cls.isdigit() else "",
        "patient_id": knee.patient_id,
        "knee_side": knee.knee_side,
        "original_split": split,
        "width": "", "height": "", "channels": "", "dtype": "",
        "file_size": path.stat().st_size,
        "format": "", "is_readable": False, "is_corrupt": True,
        "mean_pixel": "", "std_pixel": "", "min_pixel": "", "max_pixel": "",
        "aspect_ratio": "", "unique_hash": "", "perceptual_hash": "",
    }

    try:
        row["unique_hash"] = sha256_file(path)
    except OSError as exc:  # pragma: no cover
        LOG.error("cannot read bytes of %s: %s", path, exc)
        return row

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")           # promote PIL warnings to errors
            with Image.open(path) as im:
                im.load()                             # force full decode
                row["format"] = im.format or ""
                row["width"], row["height"] = im.size
                pil_mode = im.mode
                if imagehash is not None:
                    row["perceptual_hash"] = str(imagehash.phash(im.convert("L")))
                arr = np.array(im)
        row["is_readable"] = True
        row["is_corrupt"] = False
        row["channels"] = 1 if arr.ndim == 2 else arr.shape[2]
        row["dtype"] = str(arr.dtype)
        row["mean_pixel"] = round(float(arr.mean()), 4)
        row["std_pixel"] = round(float(arr.std()), 4)
        row["min_pixel"] = float(arr.min())
        row["max_pixel"] = float(arr.max())
        row["aspect_ratio"] = round(row["width"] / row["height"], 4) if row["height"] else ""
        row["_pil_mode"] = pil_mode
    except Exception as exc:  # noqa: BLE001 - we deliberately catch everything and flag it
        LOG.warning("CORRUPT/unreadable image %s: %s", path, exc)

    return row


def audit_kaggle(input_dir: Path, source: str) -> pd.DataFrame:
    paths = list(iter_images(input_dir))
    LOG.info("auditing %d images under %s", len(paths), input_dir)
    rows = []
    for i, p in enumerate(paths, 1):
        rows.append(audit_image(p, source, input_dir))
        if i % 1000 == 0:
            LOG.info("  ... %d/%d", i, len(paths))
    df = pd.DataFrame(rows)
    return df


def audit_oai_metadata(summary_csv: Path, source: str) -> pd.DataFrame:
    """The OAI repo contains NO pixel data - only metadata CSVs + code + weights.

    We still emit an audit row per knee record so the provenance is captured, with
    the image-level fields left blank and is_readable=False (no file on disk).
    """
    src = pd.read_csv(summary_csv)
    LOG.info("OAI_summary.csv: %d knee records, %d unique patients",
             len(src), src["ID"].nunique())
    out = []
    for _, r in src.iterrows():
        try:
            klg = int(float(r["KLG"]))
        except (ValueError, TypeError):
            klg = ""
        side = "R" if str(r["SIDE"]) == "1" else "L" if str(r["SIDE"]) == "2" else "UNKNOWN"
        out.append({
            "dataset_source": source,
            "file_path": f"(no pixel data in repo) {r['Folder']}",
            "filename": f"{r['ID']}_{side}_{r['Visit']}",
            "class": klg,
            "class_name": KL_CLASS_NAMES.get(klg, "") if klg != "" else "",
            "patient_id": str(r["ID"]),
            "knee_side": side,
            "original_split": str(r["Visit"]),
            "width": "", "height": "", "channels": "", "dtype": "",
            "file_size": "", "format": "DICOM(not present)",
            "is_readable": False, "is_corrupt": "",
            "mean_pixel": "", "std_pixel": "", "min_pixel": "", "max_pixel": "",
            "aspect_ratio": "", "unique_hash": "", "perceptual_hash": "",
        })
    return pd.DataFrame(out, columns=AUDIT_COLUMNS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="dataset source tag, e.g. 'kaggle' or 'oai'")
    ap.add_argument("--input-dir", type=Path, help="root folder of images (kaggle mode)")
    ap.add_argument("--oai-summary", type=Path, help="path to OAI_summary.csv (oai metadata mode)")
    ap.add_argument("--out", type=Path, required=True, help="output CSV path")
    args = ap.parse_args()

    if args.oai_summary:
        df = audit_oai_metadata(args.oai_summary, args.source)
    else:
        if not args.input_dir or not args.input_dir.exists():
            ap.error("--input-dir must exist in kaggle mode")
        df = audit_kaggle(args.input_dir, args.source)

    df = df.reindex(columns=AUDIT_COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, quoting=csv.QUOTE_MINIMAL)
    LOG.info("wrote %s (%d rows)", args.out, len(df))

    # quick console summary
    if df["is_readable"].any():
        ok = df[df["is_readable"] == True]  # noqa: E712
        LOG.info("readable=%d corrupt=%d", len(ok), int((df["is_corrupt"] == True).sum()))  # noqa: E712
        LOG.info("sizes: %s", ok.groupby(["width", "height"]).size().to_dict())
        LOG.info("channels: %s", ok["channels"].value_counts().to_dict())
        LOG.info("class counts: %s", df["class"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
