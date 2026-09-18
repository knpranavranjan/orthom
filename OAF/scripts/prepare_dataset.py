"""STEP 13 - Materialise training-ready images from the manifest splits.

    python scripts/prepare_dataset.py --config configs/preprocessing.yaml

For each requested contrast variant and each split it runs the deterministic
pipeline (read -> grayscale -> validate -> ROI -> pad -> resize -> contrast) and
writes lossless uint8 PNGs to::

    data/processed/images/<variant>/<split>/<kl>/<sample_id>.png

The pre-resize grayscale image is also saved once under
``data/interim/cleaned/<split>/<kl>/`` (spec section 15: keep original + processed).
A per-variant processed manifest is written to ``metadata/``.

Float scaling / intensity normalization / 1->3 channel replication are applied
at load time by ``src.datasets.xray_dataset`` so the on-disk copy stays lossless.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, SPLIT_NAMES, ensure_dir, get_logger  # noqa: E402
from src.preprocessing.pipeline import DeterministicPreprocessor, load_config, variants_from_config  # noqa: E402
from src.preprocessing.transforms import load_grayscale  # noqa: E402

LOG = get_logger("prepare")


def process_split(split: str, cfg: dict, pre: DeterministicPreprocessor,
                  variant: str, out_root: Path, interim_root: Path,
                  save_interim: bool) -> pd.DataFrame:
    split_csv = Path(cfg["paths"]["splits_dir"]) / f"{split}.csv"
    if not split_csv.is_absolute():
        split_csv = PROJECT_ROOT / split_csv
    out_root = out_root if out_root.is_absolute() else PROJECT_ROOT / out_root
    interim_root = interim_root if interim_root.is_absolute() else PROJECT_ROOT / interim_root
    df = pd.read_csv(split_csv, dtype={"kl_grade": int, "patient_id": str})
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"{variant}/{split}"):
        src = PROJECT_ROOT / r["original_path"]
        kl = int(r["kl_grade"])
        gray = load_grayscale(src)
        proc = pre.process_array(gray)

        out_dir = ensure_dir(out_root / variant / split / str(kl))
        out_path = out_dir / f"{r['sample_id']}.png"
        cv2.imwrite(str(out_path), proc)

        interim_path = ""
        if save_interim and variant == variants_from_config(cfg)[0]["name"]:
            idir = ensure_dir(interim_root / split / str(kl))
            ipath = idir / f"{r['sample_id']}.png"
            if not ipath.exists():
                cv2.imwrite(str(ipath), gray)
            interim_path = str(ipath.relative_to(PROJECT_ROOT).as_posix())

        rows.append({
            "sample_id": r["sample_id"], "split": split, "kl_grade": kl,
            "class_name": r["class_name"], "patient_id": r["patient_id"],
            "knee_side": r["knee_side"],
            "original_path": r["original_path"],
            "interim_path": interim_path,
            "processed_path": out_path.relative_to(PROJECT_ROOT).as_posix(),
            "variant": variant,
            "proc_h": proc.shape[0], "proc_w": proc.shape[1],
            "proc_mean": round(float(proc.mean()), 3),
            "proc_std": round(float(proc.std()), 3),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/preprocessing.yaml"))
    ap.add_argument("--input-dir", type=Path, help="override raw_root")
    ap.add_argument("--output-dir", type=Path, help="override processed_root")
    ap.add_argument("--manifest", type=Path, help="override manifest path")
    ap.add_argument("--split", choices=[*SPLIT_NAMES, "all"], default="all")
    ap.add_argument("--image-size", type=int, nargs=2, metavar=("H", "W"))
    ap.add_argument("--roi-mode", choices=["full", "center_crop", "provided"])
    ap.add_argument("--normalization", choices=["dataset_grayscale", "imagenet", "minmax", "zscore_per_image"])
    ap.add_argument("--contrast", choices=["none", "clahe", "hist_eq"],
                    help="materialise ONLY this contrast as a single variant")
    ap.add_argument("--variants", default="config",
                    help="'config' (use yaml variants) | 'all' | comma list of names")
    ap.add_argument("--no-interim", action="store_true", help="skip saving pre-resize copies")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.input_dir:
        cfg["paths"]["raw_root"] = str(args.input_dir)
    if args.output_dir:
        cfg["paths"]["processed_root"] = str(args.output_dir)
    if args.manifest:
        cfg["paths"]["manifest"] = str(args.manifest)
    if args.image_size:
        cfg["image"]["size"] = list(args.image_size)
    if args.roi_mode:
        cfg["roi"]["mode"] = args.roi_mode
    if args.normalization:
        cfg["intensity_normalization"]["mode"] = args.normalization
    cfg["seed"] = args.seed

    if args.contrast:
        variants = [{"name": {"none": "basic", "clahe": "clahe", "hist_eq": "histeq"}[args.contrast],
                     "contrast": args.contrast}]
    elif args.variants == "config":
        variants = variants_from_config(cfg)
    elif args.variants == "all":
        variants = [{"name": "basic", "contrast": "none"},
                    {"name": "clahe", "contrast": "clahe"},
                    {"name": "histeq", "contrast": "hist_eq"}]
    else:
        wanted = {v.strip() for v in args.variants.split(",")}
        variants = [v for v in variants_from_config(cfg) if v["name"] in wanted]

    splits = list(SPLIT_NAMES) if args.split == "all" else [args.split]
    out_root = Path(cfg["paths"]["processed_root"])
    interim_root = Path(cfg["paths"]["interim_root"])
    LOG.info("config: size=%s roi=%s norm=%s variants=%s splits=%s",
             cfg["image"]["size"], cfg["roi"]["mode"],
             cfg["intensity_normalization"]["mode"],
             [v["name"] for v in variants], splits)

    for v in variants:
        pre = DeterministicPreprocessor(cfg, contrast_override=v["contrast"])
        parts = []
        for s in splits:
            parts.append(process_split(s, cfg, pre, v["name"], out_root, interim_root,
                                       save_interim=not args.no_interim))
        pm = pd.concat(parts, ignore_index=True)
        if cfg["output"].get("write_processed_manifest", True):
            mpath = Path(cfg["paths"]["splits_dir"]) / f"processed_manifest_{v['name']}.csv"
            pm.to_csv(mpath, index=False)
            LOG.info("wrote %s (%d rows)", mpath, len(pm))
        LOG.info("variant '%s' done: %s", v["name"],
                 pm.groupby(["split", "kl_grade"]).size().unstack(fill_value=0).to_dict("index"))


if __name__ == "__main__":
    main()
