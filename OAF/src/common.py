"""Shared helpers for the AETHER-OA X-ray preprocessing pipeline.

Only standard scientific-Python dependencies are used here (numpy, pandas,
Pillow, OpenCV).  Nothing in this module imports torch so it can be used by the
audit / manifest / split stages that run before any deep-learning code exists.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

KAGGLE_RAW_DIR = PROJECT_ROOT / "kaggle"
OAI_REPO_DIR = PROJECT_ROOT / "OAI-KL-Grade-Classification"

DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = PROJECT_ROOT / "metadata"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIGS_DIR = PROJECT_ROOT / "configs"

PROCESSED_IMAGES_DIR = DATA_DIR / "processed" / "images"
INTERIM_DIR = DATA_DIR / "interim" / "cleaned"

# Canonical KL grade -> human readable name (Kellgren-Lawrence).
KL_CLASS_NAMES = {
    0: "Normal",
    1: "Doubtful",
    2: "Minimal",
    3: "Moderate",
    4: "Severe",
}

SPLIT_NAMES = ("train", "val", "test")
RANDOM_SEED = 42


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a single stream handler.

    Writes to STDOUT (not stderr) so `python ... | Tee-Object` / `| tee` in
    PowerShell captures the log cleanly instead of wrapping the first line in a
    NativeCommandError.
    """
    import sys
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


_FILE_LOG_FMT = logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def add_file_logger(path) -> None:
    """Attach a shared FileHandler to the root logger so every module logger also
    writes to `path`. Lets callers get a clean SIH log WITHOUT shell piping
    (avoids PowerShell wrapping native stderr as NativeCommandError)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(path.resolve()):
            return
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setFormatter(_FILE_LOG_FMT)
    fh.setLevel(logging.INFO)
    root.addHandler(fh)
    root.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
# The Kaggle "Knee Osteoarthritis Dataset with Severity Grading" ships two
# filename conventions for the very same underlying OAI baseline knees:
#   train/ val/ test/      -> "<OAI_ID><L|R>.png"     e.g. 9001695L.png
#   auto_test/             -> "<OAI_ID>_<1|2>.png"    e.g. 9003175_1.png  (1=R, 2=L)
_RE_LR = re.compile(r"^(?P<pid>\d+)(?P<side>[LR])$")
_RE_NUM = re.compile(r"^(?P<pid>\d+)_(?P<side>[12])$")

_SIDE_NUM_TO_LETTER = {"1": "R", "2": "L"}


@dataclass(frozen=True)
class KneeId:
    patient_id: str
    knee_side: str  # "L" | "R" | "UNKNOWN"

    @property
    def sample_key(self) -> str:
        """Stable per-knee identifier independent of the source naming scheme."""
        return f"{self.patient_id}_{self.knee_side}"


def parse_knee_filename(stem: str) -> KneeId:
    """Parse an OAI/Kaggle knee filename stem into (patient_id, knee_side)."""
    m = _RE_LR.match(stem)
    if m:
        return KneeId(m.group("pid"), m.group("side"))
    m = _RE_NUM.match(stem)
    if m:
        return KneeId(m.group("pid"), _SIDE_NUM_TO_LETTER[m.group("side")])
    return KneeId(stem, "UNKNOWN")


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of raw file bytes (exact-duplicate detection)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def iter_images(root: Path, exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")) -> Iterable[Path]:
    exts = {e.lower() for e in exts}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def rel_to_root(path: Path, root: Path = PROJECT_ROOT) -> str:
    """POSIX-style path relative to the project root (portable across OSes)."""
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
